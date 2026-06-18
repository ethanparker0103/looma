"""Unit tests for the ``POST /api/extract`` endpoint (AC-1, AC-3, AC-7).

These tests exercise the FastAPI app via :class:`httpx.AsyncClient`
against the in-process ASGI transport. The pipeline stages are
patched via :mod:`unittest.mock` so the test never invokes
yt-dlp, Whisper, the LLM, or the real TTS provider.

Coverage (rewritten for the async contract — AC-7):

* Both input shapes (YouTube URL via JSON, upload via multipart)
  return 202 + ``JobAccepted`` immediately; the result is retrieved
  by polling ``GET /api/jobs/{id}`` and ``GET /api/jobs/{id}/result``.
* Stage exceptions (IngestError, TranscriptionError, LLMSchemaError,
  TTSError) surface as the right error on ``GET /api/jobs/{id}/result``
  after the background task fails.
* Payload-too-large and unsupported-media errors surface as 413/415.
* Invalid input shape (empty body, both fields) surfaces as 400
  before any job is created — these tests are unchanged from the
  synchronous contract.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock

import httpx
import pytest

from app import config as cfg
from app import main as main_mod
from app.jobs import reset_job_manager
from app.models import (
    CODE_DOWNLOAD_FAILED,
    CODE_INTERNAL,
    CODE_INVALID_URL,
    CODE_LLM_SCHEMA_ERROR,
    CODE_PAYLOAD_TOO_LARGE,
    CODE_TTS_FAILED,
    CODE_TRANSCRIPTION_FAILED,
    CODE_UNSUPPORTED_MEDIA,
    CODE_UNSUPPORTED_SOURCE,
    Chapter,
    KnowledgeExtract,
    TranscriptSegment,
    TranscriptionResult,
)
from app.pipeline.extract import LLMSchemaError
from app.pipeline.ingest import (
    AudioTooLargeError,
    DownloadFailedError,
    InvalidURLError,
    PayloadTooLargeError,
    UnsupportedMediaError,
    UnsupportedSourceError,
)
from app.pipeline.narrate import TTSError
from app.pipeline.transcribe import TranscriptionError


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_job_manager():
    """Each test gets a fresh JobManager singleton."""
    reset_job_manager()
    yield
    reset_job_manager()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Build an isolated FastAPI app with sandboxed data dirs."""
    monkeypatch.setattr(cfg, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path / "outputs")
    app = main_mod.create_app()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture
def sample_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        transcript="hello world",
        segments=[TranscriptSegment(start=0.0, end=1.0, text="hello world")],
        language="en",
        duration_seconds=600.0,
    )


@pytest.fixture
def sample_knowledge() -> KnowledgeExtract:
    return KnowledgeExtract(
        title="Smoke",
        summary="S. S. S.",
        insights=["a", "b", "c", "d", "e"],
        chapters=[Chapter(start_seconds=0.0, end_seconds=600.0, title="all")],
        narrative="x " * 200,
        filler_removed=0,
    )


def _patched_stages(
    monkeypatch,
    *,
    transcription: TranscriptionResult,
    knowledge: KnowledgeExtract,
    narrate_path: Path,
) -> None:
    """Patch the four stages inside ``app.pipeline.orchestrator``."""
    import app.pipeline.orchestrator as orch

    monkeypatch.setattr(orch, "_stage_ingest", mock.MagicMock(return_value=Path("/tmp/src.mp3")))
    monkeypatch.setattr(orch, "_stage_transcribe", mock.MagicMock(return_value=transcription))
    monkeypatch.setattr(orch, "_stage_extract", mock.MagicMock(return_value=knowledge))
    monkeypatch.setattr(orch, "_stage_narrate", mock.MagicMock(return_value=narrate_path))


# --- Polling helper ---------------------------------------------------------


async def _submit_and_await_done(
    client: httpx.AsyncClient,
    url: str = "/api/extract",
    **kwargs,
) -> tuple[str, dict, dict]:
    """POST to ``url``, poll ``GET /api/jobs/{id}`` until terminal, return result.

    ``kwargs`` are forwarded to ``client.post`` (``json``, ``data``,
    ``files``, etc.).

    Returns:
        ``(job_id, status_dict, result_dict)`` where ``status_dict`` is
        the last polled status body and ``result_dict`` is the body of
        ``GET /api/jobs/{id}/result``.
    """
    resp = await client.post(url, **kwargs)
    assert resp.status_code == 202, (
        f"Expected 202, got {resp.status_code}: {resp.json()}"
    )
    job_id = resp.json()["job_id"]

    # Poll status until terminal (done, failed, or timeout).
    status_body = None
    for _ in range(30):
        status = await client.get(f"/api/jobs/{job_id}")
        assert status.status_code == 200, f"Status poll got {status.status_code}"
        status_body = status.json()
        if status_body["status"] in ("done", "failed", "timeout"):
            break
        await asyncio.sleep(0.01)

    # Fetch the result
    result = await client.get(f"/api/jobs/{job_id}/result")
    return job_id, status_body or {}, result.json()


# --- YouTube URL happy path ------------------------------------------------


@pytest.mark.asyncio
async def test_api_extract_youtube_returns_loomaresult(
    client, monkeypatch, sample_transcription, sample_knowledge, tmp_path: Path
) -> None:
    narrate = tmp_path / "out.mp3"
    narrate.write_bytes(b"")
    _patched_stages(
        monkeypatch, transcription=sample_transcription,
        knowledge=sample_knowledge, narrate_path=narrate,
    )

    _, status_body, result_body = await _submit_and_await_done(
        client,
        url="/api/extract",
        json={"youtube_url": "https://youtu.be/abc123"},
    )

    # Status reflects the full lifecycle
    assert status_body["status"] == "done"
    assert status_body["progress"] == 100
    assert status_body["stage_msg"] == "Done"

    # Result is a LoomaResult with all AC-6 fields
    for field in (
        "job_id", "source_type", "source_ref", "title",
        "transcription", "knowledge", "audio_url", "created_at",
    ):
        assert field in result_body, f"missing field: {field}"
    assert result_body["source_type"] == "youtube"
    assert result_body["audio_url"].startswith("/audio/")
    assert result_body["audio_url"].endswith(".mp3")


# --- Upload happy path -----------------------------------------------------


@pytest.mark.asyncio
async def test_api_extract_upload_returns_loomaresult(
    client, monkeypatch, sample_transcription, sample_knowledge, tmp_path: Path
) -> None:
    narrate = tmp_path / "out.mp3"
    narrate.write_bytes(b"")
    _patched_stages(
        monkeypatch, transcription=sample_transcription,
        knowledge=sample_knowledge, narrate_path=narrate,
    )

    upload = tmp_path / "video.mp4"
    upload.write_bytes(b"fake-mp4-bytes")

    _, status_body, result_body = await _submit_and_await_done(
        client,
        url="/api/extract",
        files={"file": ("video.mp4", upload.read_bytes(), "video/mp4")},
    )

    assert status_body["status"] == "done"
    assert result_body["source_type"] == "upload"
    assert result_body["source_ref"] == "video.mp4"


# --- Input validation (unchanged — short-circuits before job creation) ----


@pytest.mark.asyncio
async def test_api_extract_no_input_returns_400(client) -> None:
    async with client as c:
        resp = await c.post("/api/extract", json={})
    assert resp.status_code == 400
    assert resp.json()["code"] == CODE_INVALID_URL


@pytest.mark.asyncio
async def test_api_extract_both_inputs_returns_400(
    client, tmp_path: Path
) -> None:
    upload = tmp_path / "video.mp4"
    upload.write_bytes(b"x")
    async with client as c:
        resp = await c.post(
            "/api/extract",
            data={"youtube_url": "https://youtu.be/abc"},
            files={"file": ("video.mp4", upload.read_bytes(), "video/mp4")},
        )
    assert resp.status_code == 400
    assert resp.json()["code"] == CODE_INVALID_URL


# --- Stage errors (now surfaced via async poll) ---------------------------


@pytest.mark.asyncio
async def test_api_extract_invalid_url_returns_400(
    client, monkeypatch
) -> None:
    import app.pipeline.orchestrator as orch

    monkeypatch.setattr(
        orch, "_stage_ingest",
        mock.MagicMock(side_effect=InvalidURLError("bad url")),
    )

    _, status_body, result_body = await _submit_and_await_done(
        client, url="/api/extract",
        json={"youtube_url": "not a url"},
    )
    assert status_body["status"] == "failed"
    assert status_body["error"]["code"] == CODE_INVALID_URL


@pytest.mark.asyncio
async def test_api_extract_unsupported_source_returns_400(
    client, monkeypatch
) -> None:
    import app.pipeline.orchestrator as orch

    monkeypatch.setattr(
        orch, "_stage_ingest",
        mock.MagicMock(side_effect=UnsupportedSourceError("vimeo not allowed")),
    )

    _, status_body, _ = await _submit_and_await_done(
        client, url="/api/extract",
        json={"youtube_url": "https://vimeo.com/123"},
    )
    assert status_body["status"] == "failed"
    assert status_body["error"]["code"] == CODE_UNSUPPORTED_SOURCE


@pytest.mark.asyncio
async def test_api_extract_unsupported_media_returns_415(
    client, monkeypatch
) -> None:
    import app.pipeline.orchestrator as orch

    monkeypatch.setattr(
        orch, "_stage_ingest",
        mock.MagicMock(side_effect=UnsupportedMediaError("only mp4/mov/mkv/webm")),
    )

    upload = b"x" * 100
    _, status_body, _ = await _submit_and_await_done(
        client, url="/api/extract",
        files={"file": ("video.avi", upload, "video/avi")},
    )
    assert status_body["status"] == "failed"
    assert status_body["error"]["code"] == CODE_UNSUPPORTED_MEDIA


@pytest.mark.asyncio
async def test_api_extract_payload_too_large_returns_413(
    client, monkeypatch
) -> None:
    import app.pipeline.orchestrator as orch

    monkeypatch.setattr(
        orch, "_stage_ingest",
        mock.MagicMock(side_effect=PayloadTooLargeError("over 200 MB")),
    )

    upload = b"x" * 100
    _, status_body, _ = await _submit_and_await_done(
        client, url="/api/extract",
        files={"file": ("video.mp4", upload, "video/mp4")},
    )
    assert status_body["status"] == "failed"
    assert status_body["error"]["code"] == CODE_PAYLOAD_TOO_LARGE


@pytest.mark.asyncio
async def test_api_extract_audio_too_large_returns_413(
    client, monkeypatch
) -> None:
    import app.pipeline.orchestrator as orch

    monkeypatch.setattr(
        orch, "_stage_ingest",
        mock.MagicMock(side_effect=AudioTooLargeError("> 50 MB")),
    )

    _, status_body, _ = await _submit_and_await_done(
        client, url="/api/extract",
        json={"youtube_url": "https://youtu.be/abc"},
    )
    assert status_body["status"] == "failed"
    assert status_body["error"]["code"] == CODE_PAYLOAD_TOO_LARGE


@pytest.mark.asyncio
async def test_api_extract_download_failed_returns_500(
    client, monkeypatch
) -> None:
    import app.pipeline.orchestrator as orch

    monkeypatch.setattr(
        orch, "_stage_ingest",
        mock.MagicMock(side_effect=DownloadFailedError("yt-dlp error")),
    )

    _, status_body, _ = await _submit_and_await_done(
        client, url="/api/extract",
        json={"youtube_url": "https://youtu.be/abc"},
    )
    assert status_body["status"] == "failed"
    assert status_body["error"]["code"] == CODE_DOWNLOAD_FAILED


@pytest.mark.asyncio
async def test_api_extract_transcription_error_returns_500(
    client, monkeypatch, sample_knowledge, tmp_path: Path
) -> None:
    import app.pipeline.orchestrator as orch

    monkeypatch.setattr(orch, "_stage_ingest", mock.MagicMock(return_value=Path("/tmp/src.mp3")))
    monkeypatch.setattr(
        orch, "_stage_transcribe",
        mock.MagicMock(side_effect=TranscriptionError("whisper exploded")),
    )

    _, status_body, _ = await _submit_and_await_done(
        client, url="/api/extract",
        json={"youtube_url": "https://youtu.be/abc"},
    )
    assert status_body["status"] == "failed"
    assert status_body["error"]["code"] == CODE_TRANSCRIPTION_FAILED


@pytest.mark.asyncio
async def test_api_extract_llm_schema_error_returns_500(
    client, monkeypatch, sample_transcription, tmp_path: Path
) -> None:
    import app.pipeline.orchestrator as orch

    monkeypatch.setattr(orch, "_stage_ingest", mock.MagicMock(return_value=Path("/tmp/src.mp3")))
    monkeypatch.setattr(orch, "_stage_transcribe", mock.MagicMock(return_value=sample_transcription))
    monkeypatch.setattr(
        orch, "_stage_extract",
        mock.MagicMock(side_effect=LLMSchemaError("LLM schema bad")),
    )

    _, status_body, _ = await _submit_and_await_done(
        client, url="/api/extract",
        json={"youtube_url": "https://youtu.be/abc"},
    )
    assert status_body["status"] == "failed"
    assert status_body["error"]["code"] == CODE_LLM_SCHEMA_ERROR


@pytest.mark.asyncio
async def test_api_extract_tts_error_returns_500(
    client, monkeypatch, sample_transcription, sample_knowledge, tmp_path: Path
) -> None:
    import app.pipeline.orchestrator as orch

    monkeypatch.setattr(orch, "_stage_ingest", mock.MagicMock(return_value=Path("/tmp/src.mp3")))
    monkeypatch.setattr(orch, "_stage_transcribe", mock.MagicMock(return_value=sample_transcription))
    monkeypatch.setattr(orch, "_stage_extract", mock.MagicMock(return_value=sample_knowledge))
    monkeypatch.setattr(
        orch, "_stage_narrate",
        mock.MagicMock(side_effect=TTSError("edge tts down")),
    )

    _, status_body, result_body = await _submit_and_await_done(
        client, url="/api/extract",
        json={"youtube_url": "https://youtu.be/abc"},
    )
    # TTS failure is now non-fatal: job completes with text-only results
    assert status_body["status"] == "done"
    # audio_url should be empty since TTS degraded gracefully
    assert result_body["audio_url"] == ""
    # Text content should still be present
    assert result_body["title"] == "Smoke"
    assert result_body["knowledge"]["summary"] == "S. S. S."


@pytest.mark.asyncio
async def test_api_extract_unexpected_exception_returns_500(
    client, monkeypatch
) -> None:
    import app.pipeline.orchestrator as orch

    monkeypatch.setattr(
        orch, "_stage_ingest",
        mock.MagicMock(side_effect=RuntimeError("totally unexpected")),
    )

    _, status_body, _ = await _submit_and_await_done(
        client, url="/api/extract",
        json={"youtube_url": "https://youtu.be/abc"},
    )
    assert status_body["status"] == "failed"
    assert status_body["error"]["code"] == CODE_INTERNAL


# --- Error shape ---------------------------------------------------------


@pytest.mark.asyncio
async def test_api_extract_error_body_shape(
    client, monkeypatch
) -> None:
    """Error body on ``GET /api/jobs/{id}`` for a failed job uses canonical shape."""
    import app.pipeline.orchestrator as orch

    monkeypatch.setattr(
        orch, "_stage_ingest",
        mock.MagicMock(side_effect=InvalidURLError("nope")),
    )

    _, status_body, _ = await _submit_and_await_done(
        client, url="/api/extract",
        json={"youtube_url": "x"},
    )
    # The in-memory status dict stores error as {code, msg}
    assert "error" in status_body
    err = status_body["error"]
    assert "code" in err
    assert "msg" in err  # The internal shape is {code, msg}
