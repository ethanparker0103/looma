"""Unit tests for the async submit flow (BYOK pipeline).

Tests both ``POST /api/extract`` and ``POST /api/extract/async``,
as well as the polling endpoints ``GET /api/jobs/{id}`` and
``GET /api/jobs/{id}/result``.

The real pipeline (Whisper, yt-dlp) is never invoked —
``app.main.run_job_async`` is monkey-patched to return a fake
``TranscriptionResult`` immediately.

LLM extraction is handled on the frontend (BYOK) and is not tested here.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest import mock

import httpx
import pytest

from app import config as cfg
from app import main as main_mod
from app.jobs import (
    JobManager,
    JobStatus,
    get_job_manager,
    reset_job_manager,
)
from app.models import (
    CODE_INVALID_URL,
    CODE_JOB_NOT_READY,
    CODE_NOT_FOUND,
    TranscriptSegment,
    TranscriptionResult,
)
from app.pipeline.transcribe import TranscriptionError

# Convenience: make the JobStatus enum available for mock return values.
AsyncJobStatus = JobStatus


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_job_manager():
    reset_job_manager()
    yield
    reset_job_manager()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path / "outputs")
    app = main_mod.create_app()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture
def fake_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        transcript="hello world",
        segments=[TranscriptSegment(start=0.0, end=1.0, text="hello world")],
        language="en",
        duration_seconds=600.0,
    )


# --- Polling helper ---------------------------------------------------------


async def _submit_and_await_done(
    client: httpx.AsyncClient,
    url: str = "/api/extract/async",
    **kwargs,
) -> tuple[str, dict, dict]:
    """POST to ``url``, poll until terminal, return (job_id, status, result)."""
    resp = await client.post(url, **kwargs)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    status_body = None
    for _ in range(30):
        status = await client.get(f"/api/jobs/{job_id}")
        assert status.status_code == 200
        status_body = status.json()
        if status_body["status"] in ("done", "failed", "timeout"):
            break
        await asyncio.sleep(0.01)

    result = await client.get(f"/api/jobs/{job_id}/result")
    return job_id, status_body or {}, result.json()


# --- Happy path tests -------------------------------------------------------


class TestAsyncSubmit:
    @pytest.mark.asyncio
    async def test_post_returns_202_with_job_accepted_shape(
        self, client, fake_transcription, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            main_mod, "run_job_async",
            mock.AsyncMock(return_value=fake_transcription),
        )
        async with client as c:
            resp = await c.post(
                "/api/extract/async",
                json={"youtube_url": "https://youtu.be/dQw4w9WgXcQ"},
            )
        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        assert body["status"] in ("queued", "running")
        assert body["status_url"].startswith("/api/jobs/")
        assert body["result_url"].startswith("/api/jobs/")

    @pytest.mark.asyncio
    async def test_legacy_url_also_returns_202(
        self, client, fake_transcription, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            main_mod, "run_job_async",
            mock.AsyncMock(return_value=fake_transcription),
        )
        async with client as c:
            resp = await c.post(
                "/api/extract",
                json={"youtube_url": "https://youtu.be/abc"},
            )
        assert resp.status_code == 202

    @pytest.mark.asyncio
    async def test_post_returns_202_in_under_500ms(
        self, client, fake_transcription, monkeypatch
    ) -> None:
        async def _slow_run(*a, **kw):
            await asyncio.sleep(2.0)
            return fake_transcription

        monkeypatch.setattr(
            main_mod, "run_job_async",
            mock.AsyncMock(side_effect=_slow_run),
        )
        async with client as c:
            t0 = time.perf_counter()
            resp = await c.post(
                "/api/extract/async",
                json={"youtube_url": "https://youtu.be/abc"},
            )
        elapsed = time.perf_counter() - t0
        assert resp.status_code == 202
        assert elapsed < 0.5, f"handler took {elapsed:.3f}s (should be < 0.5)"

    @pytest.mark.asyncio
    async def test_legacy_url_under_500ms(
        self, client, fake_transcription, monkeypatch
    ) -> None:
        import asyncio

        async def _slow_run(*a, **kw):
            await asyncio.sleep(2.0)
            return fake_transcription

        monkeypatch.setattr(
            main_mod, "run_job_async",
            mock.AsyncMock(side_effect=_slow_run),
        )
        async with client as c:
            t0 = time.perf_counter()
            resp = await c.post(
                "/api/extract",
                json={"youtube_url": "https://youtu.be/abc"},
            )
        elapsed = time.perf_counter() - t0
        assert resp.status_code == 202
        assert elapsed < 0.5

    @pytest.mark.asyncio
    async def test_full_async_flow_returns_transcription(
        self, client, fake_transcription, monkeypatch
    ) -> None:
        """End-to-end: submit → poll → result returns transcription."""
        monkeypatch.setattr(
            main_mod, "run_job_async",
            mock.AsyncMock(return_value=fake_transcription),
        )
        _, status_body, result_body = await _submit_and_await_done(
            client, json={"youtube_url": "https://youtu.be/abc"},
        )
        assert status_body["status"] == "done"
        assert status_body["progress"] == 100
        # Result should be the CodeExtractResponse with transcription data
        assert "transcription" in result_body
        assert "segments" in result_body
        assert result_body["language"] == "en"
        assert result_body["duration_seconds"] == 600.0


# --- Status polling tests ---------------------------------------------------


class TestStatusPolling:
    @pytest.mark.asyncio
    async def test_get_status_returns_in_memory_state(
        self, client, fake_transcription, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            main_mod, "run_job_async",
            mock.AsyncMock(return_value=fake_transcription),
        )
        async with client as c:
            submit = await c.post(
                "/api/extract/async",
                json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
            )
            job_id = submit.json()["job_id"]

            status = await c.get(f"/api/jobs/{job_id}")
            assert status.status_code == 200
            sbody = status.json()
            assert sbody["id"] == job_id
            assert sbody["status"] in ("queued", "downloading", "transcribing", "done")

    @pytest.mark.asyncio
    async def test_get_status_returns_full_contract_shape(
        self, client, fake_transcription, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            main_mod, "run_job_async",
            mock.AsyncMock(return_value=fake_transcription),
        )
        async with client as c:
            submit = await c.post(
                "/api/extract/async",
                json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
            )
            job_id = submit.json()["job_id"]

            status = await c.get(f"/api/jobs/{job_id}")
            assert status.status_code == 200
            sbody = status.json()

        assert "id" in sbody
        assert "status" in sbody
        assert "progress" in sbody
        assert "stage_msg" in sbody
        assert "source_ref" in sbody
        assert "source_type" in sbody
        assert "created_at" in sbody
        assert sbody["id"] == job_id
        # With dedup, source_ref is the stable video ID
        assert sbody["source_ref"] == "dQw4w9WgXcQ"
        assert sbody["source_type"] == "youtube"
        assert sbody["status"] in ("queued", "downloading", "transcribing", "done")

    @pytest.mark.asyncio
    async def test_get_status_returns_404_for_unknown_job(self, client) -> None:
        async with client as c:
            resp = await c.get("/api/jobs/no-such-job")
        assert resp.status_code == 404
        assert "error" in resp.json()


# --- Result endpoint tests --------------------------------------------------


class TestResultEndpoint:
    @pytest.mark.asyncio
    async def test_get_result_returns_409_when_not_done(
        self, client, fake_transcription, monkeypatch
    ) -> None:
        """Result endpoint returns 409 while the pipeline is still running."""
        import asyncio

        async def _slow_run(*a, **kw):
            await asyncio.sleep(30)
            return fake_transcription

        monkeypatch.setattr(
            main_mod, "run_job_async",
            mock.AsyncMock(side_effect=_slow_run),
        )
        async with client as c:
            submit = await c.post(
                "/api/extract/async",
                json={"youtube_url": "https://youtu.be/abc"},
            )
            job_id = submit.json()["job_id"]

            await asyncio.sleep(0.05)
            result = await c.get(f"/api/jobs/{job_id}/result")
        assert result.status_code == 409
        body = result.json()
        assert body.get("code") == CODE_JOB_NOT_READY

    @pytest.mark.asyncio
    async def test_get_result_returns_transcription_when_done(
        self, client, fake_transcription, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            main_mod, "run_job_async",
            mock.AsyncMock(return_value=fake_transcription),
        )
        _, _, result_body = await _submit_and_await_done(
            client, json={"youtube_url": "https://youtu.be/abc"},
        )
        assert result_body["duration_seconds"] == 600.0
        assert result_body["language"] == "en"

    @pytest.mark.asyncio
    async def test_get_result_returns_500_when_pipeline_fails(
        self, client, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            main_mod, "run_job_async",
            mock.AsyncMock(side_effect=TranscriptionError("whisper crashed")),
        )
        _, status_body, result_body = await _submit_and_await_done(
            client, json={"youtube_url": "https://youtu.be/abc"},
        )
        assert status_body["status"] == "failed"
        assert "whisper" in status_body.get("stage_msg", "").lower()


# --- Invalid input tests ----------------------------------------------------


class TestInvalidInput:
    @pytest.mark.asyncio
    async def test_empty_body_returns_400(self, client) -> None:
        async with client as c:
            resp = await c.post(
                "/api/extract/async",
                json={},
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_url_returns_400(self, client) -> None:
        async with client as c:
            resp = await c.post(
                "/api/extract/async",
                json={"not_a_url": 42},
            )
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body

    @pytest.mark.asyncio
    async def test_bad_content_type_returns_400(self, client) -> None:
        async with client as c:
            resp = await c.post(
                "/api/extract/async",
                content=b"not json",
                headers={"Content-Type": "text/plain"},
            )
        assert resp.status_code == 400
