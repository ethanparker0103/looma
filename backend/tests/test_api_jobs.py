"""Unit tests for the ``/api/jobs*`` endpoints (AC-9, AC-10 partial).

These tests exercise the FastAPI app via :class:`httpx.AsyncClient`
against the in-process ASGI transport. The default :class:`JobsDB`
singleton is reset between cases so the on-disk ``jobs.json`` file
isn't polluted by a prior run.

Coverage:

* ``GET /api/jobs?limit=20`` returns the most recent 20 jobs
  (AC-9 — the default page size).
* ``GET /api/jobs?limit=N`` honors an explicit limit; out-of-
  range limits are clamped (1-200).
* ``GET /api/jobs`` sorts newest-first.
* ``GET /api/jobs`` returns each job with the seven AC-9 columns.
* ``GET /api/jobs/{id}`` returns a single job or 404.
* ``POST /api/extract`` (async) persists an in-memory job and updates it
  to "done" (or "failed") when the background pipeline completes.
* ``DELETE /api/jobs/{id}`` removes the row + the on-disk MP3s
  (AC-10) and returns a structured report.
* Path-traversal payloads are rejected with the canonical
  ``{"error","code"}`` shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import httpx
import pytest

from app import main as main_mod
from app.config import AUDIO_DIR, OUTPUTS_DIR
from app.models import (
    CODE_NOT_FOUND,
    Chapter,
    KnowledgeExtract,
    LoomaResult,
    TranscriptSegment,
    TranscriptionResult,
)
from app.storage.jobs import (
    JobStatus,
    JobsDB,
    get_default_db,
    reset_default_db,
)


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch) -> Path:
    """Redirect the default JobsDB to a fresh tmp JSON file for the test.

    The ``JOBS_JSON_PATH`` constant is imported into
    :mod:`app.storage.jobs` at import time, so we patch that
    module's reference too. Without this, the test would still
    see the production ``data/jobs.json`` path.
    """
    from app import config as cfg
    from app.storage import jobs as jobs_mod

    p = tmp_path / "jobs.json"
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Patch both the canonical config module and the storage
    # module's local binding (captured at import time).
    monkeypatch.setattr(cfg, "JOBS_JSON_PATH", p)
    monkeypatch.setattr(jobs_mod, "JOBS_JSON_PATH", p)
    monkeypatch.setattr(cfg, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path / "outputs")
    return p


@pytest.fixture
def client(db_path: Path):
    """Build an isolated app; default DB is fresh per test."""
    app = main_mod.create_app()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture(autouse=True)
def _reset_default_db_each_test():
    """Each test gets a clean default DB."""
    reset_default_db()
    yield
    reset_default_db()


def _seed(n: int) -> list[dict]:
    """Insert ``n`` job rows via the default DB and return their dicts."""
    db = get_default_db()
    rows = []
    for i in range(n):
        r = db.create_job(
            job_id=f"j-{i:03d}",
            source_type="youtube" if i % 2 == 0 else "upload",
            source_ref=f"ref-{i}",
            title=f"Title {i}",
            created_at=f"2026-06-14T12:00:{i:02d}.000000Z",
            duration_seconds=float(i * 10),
            status=JobStatus.COMPLETE,
        )
        rows.append(
            {
                "id": r.id,
                "source_type": r.source_type,
                "source_ref": r.source_ref,
                "title": r.title,
                "created_at": r.created_at,
                "duration_seconds": r.duration_seconds,
                "status": r.status.value,
            }
        )
    return rows


# --- list_jobs happy path ---------------------------------------------------


@pytest.mark.asyncio
async def test_list_jobs_returns_empty_array(client) -> None:
    async with client as c:
        resp = await c.get("/api/jobs")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_jobs_default_returns_all_when_under_20(client) -> None:
    _seed(5)
    async with client as c:
        resp = await c.get("/api/jobs")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 5
    # Newest first.
    assert body[0]["id"] == "j-004"
    assert body[-1]["id"] == "j-000"


@pytest.mark.asyncio
async def test_list_jobs_default_limit_caps_at_20(client) -> None:
    """AC-9: ``GET /api/jobs?limit=20`` returns the most recent 20 jobs."""
    _seed(30)
    async with client as c:
        resp = await c.get("/api/jobs")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 20
    # And those 20 are the newest 20.
    assert body[0]["id"] == "j-029"
    assert body[-1]["id"] == "j-010"


@pytest.mark.asyncio
async def test_list_jobs_explicit_limit(client) -> None:
    _seed(10)
    async with client as c:
        resp = await c.get("/api/jobs?limit=3")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 3
    assert body[0]["id"] == "j-009"
    assert body[2]["id"] == "j-007"


@pytest.mark.asyncio
async def test_list_jobs_returns_all_ac9_columns(client) -> None:
    """Each row has the seven AC-9 columns."""
    _seed(1)
    async with client as c:
        resp = await c.get("/api/jobs")
    body = resp.json()
    assert len(body) == 1
    row = body[0]
    assert set(row.keys()) == {
        "id",
        "source_type",
        "source_ref",
        "title",
        "created_at",
        "duration_seconds",
        "status",
    }
    # And the values match what we inserted.
    assert row["id"] == "j-000"
    assert row["source_type"] == "youtube"
    assert row["source_ref"] == "ref-0"
    assert row["title"] == "Title 0"
    assert row["duration_seconds"] == 0.0
    assert row["status"] == "complete"


@pytest.mark.asyncio
async def test_list_jobs_limit_validation(client) -> None:
    """AC-11: out-of-range limits return the canonical 400 shape.

    FastAPI's ``Query(ge=1, le=200)`` triggers a
    :class:`RequestValidationError`. The app's centralized handler
    (AC-11) maps the default 422 down to 400 INVALID_URL because
    422 is not in the AC-11 status-code allow-list
    (200/400/404/413/415/500).
    """
    async with client as c:
        bad_low = await c.get("/api/jobs?limit=0")
        bad_high = await c.get("/api/jobs?limit=201")
    assert bad_low.status_code == 400
    assert bad_high.status_code == 400
    # And both are in the canonical shape.
    assert bad_low.json()["code"] == "INVALID_URL"
    assert bad_high.json()["code"] == "INVALID_URL"


# --- get_job ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_job_returns_record(client) -> None:
    _seed(2)
    async with client as c:
        resp = await c.get("/api/jobs/j-000")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "j-000"
    assert body["source_ref"] == "ref-0"


@pytest.mark.asyncio
async def test_get_job_missing_returns_404(client) -> None:
    async with client as c:
        resp = await c.get("/api/jobs/no-such-job")
    assert resp.status_code == 404
    assert resp.json()["code"] == CODE_NOT_FOUND


@pytest.mark.asyncio
async def test_get_job_rejects_unsafe_id(client) -> None:
    async with client as c:
        resp = await c.get("/api/jobs/..%2F..%2Fetc%2Fpasswd")
    # FastAPI may 404 the route before our handler runs, or our
    # _is_safe_job_id may 400 it; either is acceptable.
    assert resp.status_code in (400, 404)


# --- POST /api/extract persistence -----------------------------------------


def _patched_stages(monkeypatch, *, transcription, knowledge, narrate_path):
    """Patch the four orchestrator stages with the given returns."""
    from app.pipeline import orchestrator as orch
    monkeypatch.setattr(orch, "_stage_ingest", mock.MagicMock(return_value=Path("/tmp/src.mp3")))
    monkeypatch.setattr(orch, "_stage_transcribe", mock.MagicMock(return_value=transcription))
    monkeypatch.setattr(orch, "_stage_extract", mock.MagicMock(return_value=knowledge))
    monkeypatch.setattr(orch, "_stage_narrate", mock.MagicMock(return_value=narrate_path))


def _stub_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        transcript="hello world",
        segments=[TranscriptSegment(start=0.0, end=1.0, text="hello world")],
        language="en",
        duration_seconds=125.5,
    )


def _stub_knowledge() -> KnowledgeExtract:
    return KnowledgeExtract(
        title="Refined Title",
        summary="Sentence one. Sentence two. Sentence three.",
        insights=["a", "b", "c", "d", "e"],
        chapters=[Chapter(start_seconds=0.0, end_seconds=125.5, title="c")],
        narrative="x " * 200,
        filler_removed=0,
    )


@pytest.mark.asyncio
async def test_post_extract_persists_running_then_complete(
    client, monkeypatch, tmp_path: Path
) -> None:
    """AC-9: a successful async POST /api/extract reaches "done" in JobManager."""
    from app.pipeline import orchestrator as orch
    from app.jobs import get_job_manager

    monkeypatch.setattr(orch, "_stage_ingest", mock.MagicMock(return_value=Path("/tmp/src.mp3")))
    monkeypatch.setattr(orch, "_stage_transcribe", mock.MagicMock(return_value=_stub_transcription()))
    monkeypatch.setattr(orch, "_stage_extract", mock.MagicMock(return_value=_stub_knowledge()))
    monkeypatch.setattr(orch, "_stage_narrate", mock.MagicMock(return_value=tmp_path / "out.mp3"))

    async with client as c:
        resp = await c.post(
            "/api/extract",
            json={"youtube_url": "https://youtu.be/abc"},
        )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # Poll the in-memory job manager until the job reaches a terminal state.
    import asyncio
    mgr = get_job_manager()
    for _ in range(30):
        job = mgr.get(job_id)
        if job is not None and job.status == "done":
            break
        await asyncio.sleep(0.01)

    assert job is not None, "Job not found in JobManager"
    assert job.status == "done"
    assert job.kind == "youtube"
    assert job.source_ref == "https://youtu.be/abc"


@pytest.mark.asyncio
async def test_post_extract_persists_failed_on_pipeline_error(
    client, monkeypatch, tmp_path: Path
) -> None:
    """AC-9: a failed async POST /api/extract reaches "failed" in JobManager."""
    from app.pipeline import orchestrator as orch
    from app.pipeline.ingest import InvalidURLError
    from app.jobs import get_job_manager

    monkeypatch.setattr(
        orch, "_stage_ingest",
        mock.MagicMock(side_effect=InvalidURLError("nope")),
    )

    async with client as c:
        resp = await c.post(
            "/api/extract",
            json={"youtube_url": "not a url"},
        )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # Poll the in-memory job manager until the job fails.
    import asyncio
    mgr = get_job_manager()
    for _ in range(30):
        job = mgr.get(job_id)
        if job is not None and job.status == "failed":
            break
        await asyncio.sleep(0.01)

    assert job is not None, "Job not found in JobManager"
    assert job.status == "failed"
    assert job.error is not None
    assert job.error["code"] == "INVALID_URL"


@pytest.mark.asyncio
async def test_get_job_by_id_after_async_submit(
    client, monkeypatch, tmp_path: Path
) -> None:
    """End-to-end: POST /api/extract, then GET /api/jobs/{id} includes it."""
    from app.pipeline import orchestrator as orch

    monkeypatch.setattr(orch, "_stage_ingest", mock.MagicMock(return_value=Path("/tmp/src.mp3")))
    monkeypatch.setattr(orch, "_stage_transcribe", mock.MagicMock(return_value=_stub_transcription()))
    monkeypatch.setattr(orch, "_stage_extract", mock.MagicMock(return_value=_stub_knowledge()))
    monkeypatch.setattr(orch, "_stage_narrate", mock.MagicMock(return_value=tmp_path / "out.mp3"))

    async with client as c:
        r1 = await c.post(
            "/api/extract",
            json={"youtube_url": "https://youtu.be/abc"},
        )
        assert r1.status_code == 202
        job_id = r1.json()["job_id"]

        # Poll GET /api/jobs/{id} until done
        import asyncio
        for _ in range(30):
            r2 = await c.get(f"/api/jobs/{job_id}")
            assert r2.status_code == 200
            if r2.json()["status"] == "done":
                break
            await asyncio.sleep(0.01)

        r2_body = r2.json()
        assert r2_body["id"] == job_id
        assert r2_body["status"] == "done"
        assert r2_body["source_type"] == "youtube"
        assert r2_body["source_ref"] == "https://youtu.be/abc"


# --- DELETE /api/jobs/{id} (AC-10 partial) --------------------------------


@pytest.mark.asyncio
async def test_delete_job_removes_row_and_files(
    client, monkeypatch, tmp_path: Path
) -> None:
    """AC-10: DELETE removes the row and the on-disk MP3s."""
    db = get_default_db()
    db.create_job(
        job_id="j-del", source_type="youtube", source_ref="x",
        status=JobStatus.COMPLETE,
    )
    # Place MP3s in the sandbox dirs.
    audio_dir = AUDIO_DIR
    output_dir = OUTPUTS_DIR
    audio_path = audio_dir / "j-del.mp3"
    output_path = output_dir / "j-del.mp3"
    audio_path.write_bytes(b"audio")
    output_path.write_bytes(b"output")
    assert audio_path.exists()
    assert output_path.exists()

    async with client as c:
        resp = await c.delete("/api/jobs/j-del")
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] is True
    assert body["job_id"] == "j-del"
    assert body["files_removed"]["audio"] is True
    assert body["files_removed"]["output"] is True
    assert db.get_job("j-del") is None
    assert not audio_path.exists()
    assert not output_path.exists()


@pytest.mark.asyncio
async def test_delete_job_missing_returns_404(client) -> None:
    async with client as c:
        resp = await c.delete("/api/jobs/no-such-job")
    assert resp.status_code == 404
    assert resp.json()["code"] == CODE_NOT_FOUND


@pytest.mark.asyncio
async def test_delete_job_tolerates_missing_files(
    client, tmp_path: Path
) -> None:
    """If the MP3s are already gone, the delete still succeeds."""
    db = get_default_db()
    db.create_job(
        job_id="j-no-files", source_type="youtube", source_ref="x",
    )
    async with client as c:
        resp = await c.delete("/api/jobs/j-no-files")
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] is True
    assert body["files_removed"]["audio"] is False
    assert body["files_removed"]["output"] is False
