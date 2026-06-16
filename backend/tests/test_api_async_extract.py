"""Unit tests for the async submit flow (AC-1, AC-2, AC-3, AC-4, AC-6).

These tests exercise both ``POST /api/extract`` and
``POST /api/extract/async`` — which now share the same
:func:`app.main._submit_async_job` helper — as well as the polling
endpoints ``GET /api/jobs/{id}`` and ``GET /api/jobs/{id}/result``.

The real pipeline (Whisper, LLM, yt-dlp, Edge TTS) is **never**
invoked. We monkey-patch :func:`app.main.run_job_async` so the
background task returns a fake ``LoomaResult`` immediately (or
sleeps when we want to verify the timeout / under-500-ms contract).

Coverage:

1. ``POST`` to either URL returns 202 + ``JobAccepted`` body with
   ``job_id``, ``status_url``, ``result_url``.
2. The wall-clock of the handler is under 500 ms even when the
   mocked pipeline sleeps for 60 seconds.
3. ``GET /api/jobs/{id}`` returns the in-memory ``JobState``.
4. ``GET /api/jobs/{id}/result`` returns 409 while the job is
   still running and 200 once done.
5. The legacy ``/api/extract`` URL behaves identically to the
   explicit ``/api/extract/async`` URL.
6. Invalid inputs (empty, no URL, both URL+file) return 400
   synchronously without creating a job.
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
    JobState,
    JobStatus,
    get_job_manager,
    reset_job_manager,
)
from app.models import (
    CODE_INVALID_URL,
    CODE_JOB_NOT_READY,
    CODE_NOT_FOUND,
    Chapter,
    KnowledgeExtract,
    LoomaResult,
    TranscriptSegment,
    TranscriptionResult,
)

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
def fake_result() -> dict:
    """A "stub" LoomaResult dict that passable for the real thing.

    The async job stores the result as a plain dict
    (``job.result = result.model_dump(mode='json')``), so we
    build the same shape here.
    """
    return LoomaResult(
        job_id="fake",
        source_type="youtube",
        source_ref="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="Never Gonna Give You Up",
        transcription=TranscriptionResult(
            transcript="We're no strangers to love",
            segments=[TranscriptSegment(start=0.0, end=3.5, text="We're no strangers to love")],
            language="en",
            duration_seconds=3.5,
        ),
        knowledge=KnowledgeExtract(
            title="Never Gonna Give You Up",
            summary="A classic Rick Astley song.",
            insights=["a", "b", "c", "d", "e"],
            chapters=[Chapter(start_seconds=0.0, end_seconds=3.5, title="Intro")],
            narrative="We're no strangers to love. " * 50,
            filler_removed=0,
        ),
        audio_url="/audio/fake.mp3",
        created_at="2024-01-01T00:00:00.000000Z",
    ).model_dump(mode="json")


# --- AC-1: POST returns 202 + job_id -------------------------------------


@pytest.mark.asyncio
async def test_post_returns_202_with_job_id(
    client, monkeypatch, fake_result
) -> None:
    """Body shape: status=queued, status_url and result_url are well-formed."""
    monkeypatch.setattr(
        main_mod, "run_job_async",
        mock.AsyncMock(return_value=LoomaResult(**fake_result)),
    )

    async with client as c:
        resp = await c.post(
            "/api/extract/async",
            json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )

    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "queued"
    assert body["status_url"] == f"/api/jobs/{body['job_id']}"
    assert body["result_url"] == f"/api/jobs/{body['job_id']}/result"
    # job_id should be a non-empty hex string
    assert len(body["job_id"]) > 0


@pytest.mark.asyncio
async def test_post_returns_202_in_under_500ms(
    client, monkeypatch, fake_result
) -> None:
    """Wall-clock stays under 500 ms even with a 60-second mocked pipeline."""
    monkeypatch.setattr(
        main_mod, "run_job_async",
        mock.AsyncMock(side_effect=lambda *a, **kw: asyncio.sleep(60)),
    )

    async with client as c:
        start = time.perf_counter()
        resp = await c.post(
            "/api/extract/async",
            json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        elapsed = time.perf_counter() - start

    assert resp.status_code == 202
    assert elapsed < 0.5, (
        f"Handler took {elapsed:.3f}s, expected < 0.5s"
    )


# --- AC-1 + AC-3: legacy /api/extract also returns 202 -------------------


@pytest.mark.asyncio
async def test_legacy_sync_url_also_returns_202(
    client, monkeypatch, fake_result
) -> None:
    """``POST /api/extract`` (legacy) returns the same 202 + job_id shape."""
    monkeypatch.setattr(
        main_mod, "run_job_async",
        mock.AsyncMock(return_value=LoomaResult(**fake_result)),
    )

    async with client as c:
        resp = await c.post(
            "/api/extract",
            json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )

    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "queued"
    assert body["status_url"].startswith("/api/jobs/")
    assert body["result_url"].startswith("/api/jobs/")


@pytest.mark.asyncio
async def test_legacy_url_under_500ms(
    client, monkeypatch
) -> None:
    """The legacy URL also completes in under 500 ms."""
    monkeypatch.setattr(
        main_mod, "run_job_async",
        mock.AsyncMock(side_effect=lambda *a, **kw: asyncio.sleep(60)),
    )

    async with client as c:
        start = time.perf_counter()
        resp = await c.post(
            "/api/extract",
            json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        elapsed = time.perf_counter() - start

    assert resp.status_code == 202
    assert elapsed < 0.5, (
        f"Legacy handler took {elapsed:.3f}s, expected < 0.5s"
    )


# --- AC-2: polling returns in-memory state --------------------------------


@pytest.mark.asyncio
async def test_get_status_returns_in_memory_state(
    client, monkeypatch, fake_result
) -> None:
    """``GET /api/jobs/{id}`` returns the in-memory status."""
    monkeypatch.setattr(
        main_mod, "run_job_async",
        mock.AsyncMock(return_value=LoomaResult(**fake_result)),
    )

    async with client as c:
        # 1) Submit a job
        submit = await c.post(
            "/api/extract/async",
            json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        job_id = submit.json()["job_id"]

        # 2) Poll status while it's still queued (the mocked pipeline
        #    completes instantly, so we may already be done — but we
        #    assert the response is *some* JobState shape)
        status = await c.get(f"/api/jobs/{job_id}")
        assert status.status_code == 200
        sbody = status.json()
        assert sbody["id"] == job_id
        assert sbody["status"] in ("queued", "downloading", "transcribing", "done")
        assert "progress" in sbody
        assert "stage_msg" in sbody
        # Fields from the in-memory state, not the SQLite row
        assert "created_at" in sbody
        assert "updated_at" in sbody


@pytest.mark.asyncio
async def test_get_result_returns_409_until_done(
    client, monkeypatch, fake_result
) -> None:
    """``GET /api/jobs/{id}/result`` returns 409 while running, then 200."""
    # Make the pipeline block on a barrier so we can observe the
    # in-flight status.
    barrier = asyncio.Event()

    async def _slow_pipeline(*a, **kw) -> LoomaResult:
        await barrier.wait()  # Block until we lift it
        return LoomaResult(**fake_result)

    monkeypatch.setattr(
        main_mod, "run_job_async",
        mock.AsyncMock(side_effect=_slow_pipeline),
    )

    async with client as c:
        # 1) Submit a job
        submit = await c.post(
            "/api/extract/async",
            json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        assert submit.status_code == 202
        job_id = submit.json()["job_id"]

        # 2) Poll result while the pipeline is still blocked — expect 409
        result = await c.get(f"/api/jobs/{job_id}/result")
        assert result.status_code == 409, (
            f"Expected 409 while job running, got {result.status_code}: "
            f"{result.json()}"
        )
        rbody = result.json()
        assert rbody["code"] == CODE_JOB_NOT_READY

        # 3) Unblock the pipeline
        barrier.set()
        # Give the event loop a tick to process the done callback
        await asyncio.sleep(0)

        # 4) Now the result should be available
        result2 = await c.get(f"/api/jobs/{job_id}/result")
        assert result2.status_code == 200
        r2body = result2.json()
        assert "title" in r2body
        assert r2body["title"] == fake_result["title"]


# --- AC-2: polling lifecycle edge cases ----------------------------------


@pytest.mark.asyncio
async def test_get_status_returns_404_for_unknown_job(client) -> None:
    """``GET /api/jobs/{id}`` returns 404 for a non-existent job."""
    async with client as c:
        resp = await c.get("/api/jobs/nonexistent-job-id")
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == CODE_NOT_FOUND


@pytest.mark.asyncio
async def test_get_result_returns_404_for_unknown_job(client) -> None:
    """``GET /api/jobs/{id}/result`` returns 404 for a non-existent job."""
    async with client as c:
        resp = await c.get("/api/jobs/nonexistent-job-id/result")
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == CODE_NOT_FOUND


@pytest.mark.asyncio
async def test_get_status_returns_full_contract_shape(
    client, monkeypatch, fake_result
) -> None:
    """Status response includes all AC-2 fields: id, status, progress, stage_msg,
    source_ref, source_type, created_at, updated_at."""
    monkeypatch.setattr(
        main_mod, "run_job_async",
        mock.AsyncMock(return_value=LoomaResult(**fake_result)),
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

    # Required fields per AC-2 contract
    assert "id" in sbody
    assert "status" in sbody
    assert "progress" in sbody
    assert "stage_msg" in sbody
    assert "source_ref" in sbody, "status response missing source_ref"
    assert "source_type" in sbody, "status response missing source_type"
    assert "created_at" in sbody
    assert "updated_at" in sbody

    # Values should be sensible
    assert sbody["id"] == job_id
    assert sbody["source_ref"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert sbody["source_type"] == "youtube"
    # The pipeline mock finishes instantly, so status will be "done"
    assert sbody["status"] in ("queued", "downloading", "transcribing", "done")
    assert isinstance(sbody["progress"], int)
    assert 0 <= sbody["progress"] <= 100
    assert sbody["stage_msg"]


@pytest.mark.asyncio
async def test_get_result_returns_500_for_failed_job(
    client, monkeypatch
) -> None:
    """``GET /api/jobs/{id}/result`` returns 500 with canonical shape for failed."""
    monkeypatch.setattr(
        main_mod, "run_job_async",
        mock.AsyncMock(side_effect=RuntimeError("pipeline crashed")),
    )

    async with client as c:
        submit = await c.post(
            "/api/extract/async",
            json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        assert submit.status_code == 202
        job_id = submit.json()["job_id"]

        # Wait for the background task to fail
        import time
        deadline = time.time() + 15
        while time.time() < deadline:
            result = await c.get(f"/api/jobs/{job_id}/result")
            if result.status_code != 409:
                break
            await asyncio.sleep(0.05)

    assert result.status_code == 500, (
        f"Expected 500 for failed job, got {result.status_code}: {result.json()}"
    )
    rbody = result.json()
    # The error body must have the canonical AC-11 shape
    assert "error" in rbody
    assert "code" in rbody
    assert rbody["code"] == "INTERNAL_ERROR"


@pytest.mark.asyncio
async def test_get_result_returns_408_for_timeout_job(
    client, monkeypatch
) -> None:
    """``GET /api/jobs/{id}/result`` returns 408 for a timed-out job."""
    # Create a job directly in the manager with TIMEOUT status.
    # We use the API to submit a job (to get a valid job_id), then
    # force it to TIMEOUT via the manager.
    monkeypatch.setattr(
        main_mod, "run_job_async",
        mock.AsyncMock(side_effect=lambda *a, **kw: asyncio.sleep(3600)),
    )

    async with client as c:
        submit = await c.post(
            "/api/extract/async",
            json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        assert submit.status_code == 202
        job_id = submit.json()["job_id"]

        # Manually force the in-memory state to TIMEOUT (as the
        # watchdog would) so we don't have to wait for it.
        mgr = get_job_manager()
        job = mgr.get(job_id)
        assert job is not None
        mgr.update(
            job_id,
            status=JobStatus.TIMEOUT,
            stage_msg="Timed out",
            error={"code": "TIMEOUT", "msg": "Job exceeded the time budget."},
        )
        # Cancel the background task so it doesn't overwrite our state
        mgr.detach_task(job_id)

        # Now check the result endpoint
        result = await c.get(f"/api/jobs/{job_id}/result")

    assert result.status_code == 408, (
        f"Expected 408 for timed-out job, got {result.status_code}: {result.json()}"
    )
    rbody = result.json()
    assert "error" in rbody
    assert "code" in rbody


# --- AC-1: invalid input returns 400 synchronously -----------------------


@pytest.mark.asyncio
async def test_invalid_input_returns_400_synchronously(
    client, monkeypatch
) -> None:
    """Bad URL, missing input, both-inputs-set: all return 400 before any job."""
    # Avoid accidental pipeline calls
    monkeypatch.setattr(
        main_mod, "run_job_async",
        mock.AsyncMock(),
    )

    async with client as c:
        # 1) Empty body
        resp1 = await c.post("/api/extract/async", json={})
        assert resp1.status_code == 400
        assert resp1.json()["code"] == CODE_INVALID_URL

        # 2) Both youtube_url and file set
        resp2 = await c.post(
            "/api/extract/async",
            data={"youtube_url": "https://youtu.be/abc"},
            files={"file": ("video.mp4", b"x", "video/mp4")},
        )
        assert resp2.status_code == 400
        assert resp2.json()["code"] == CODE_INVALID_URL

        # 3) No input at all (no JSON, no file)
        resp3 = await c.post("/api/extract/async", data={})
        assert resp3.status_code == 400

    # Verify that no job was created (the manager is still empty)
    mgr = get_job_manager()
    assert len(mgr._jobs) == 0, (
        f"Expected no jobs, got {len(mgr._jobs)}"
    )
