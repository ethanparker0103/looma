"""FastAPI application (AC-7, AC-8, AC-9, AC-10, AC-11, async flow).

This module ships:

* ``GET /audio/{job_id}.mp3`` — serves the TTS-generated narration
  MP3 from ``data/outputs/<job_id>.mp3`` (AC-7).
* ``POST /api/extract`` *and* ``POST /api/extract/async`` — both
  accept either a JSON body with ``youtube_url`` or a multipart
  upload with ``file`` (AC-2 + AC-3). Both submit the pipeline as a
  background ``asyncio`` task and return ``202 Accepted`` + ``job_id``
  in milliseconds (AC-1). The legacy ``/api/extract`` URL is
  preserved as an alias so any external script / curl that targets
  it still gets a fast 202 (AC-3) instead of a 524. The two routes
  share a single :func:`_submit_async_job` helper (AC-4).
* ``GET /api/jobs?limit=20`` — returns the most recent N jobs
  as JSON (AC-9). Default page size 20, max 200.
* ``GET /api/jobs/{job_id}`` — returns the in-memory async job
  state if present, else falls back to the JSON-backed file.
* ``GET /api/jobs/{job_id}/result`` — returns the
  :class:`~app.models.LoomaResult` for an async job whose status is
  ``done``; 409 ``JOB_NOT_READY`` while the job is still running.
* ``DELETE /api/jobs/{job_id}`` — removes the job row *and* its
  on-disk MP3s (AC-10).
* ``GET /`` — serves the single-page frontend (AC-8).
* ``GET /healthz`` — minimal liveness probe; always 200.

Startup guards (AC-14)
----------------------
Before the app starts serving traffic we verify that
``ffmpeg``/``ffprobe`` is on PATH and that at least one LLM API key
is set (``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``). If either is
missing the process exits non-zero with a one-line actionable error
so the user can fix their environment without reading the source.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from .config import (
    DEFAULT_JOBS_LIMIT,
    JOB_SWEEP_INTERVAL_SECONDS,
    JOB_TTL_SECONDS,
    JOB_WATCHDOG_INTERVAL_SECONDS,
    MAX_CONCURRENT_JOBS,
    OUTPUTS_DIR,
)
from .models import (
    CODE_NOT_FOUND,
    CODE_INVALID_URL,
    CODE_PAYLOAD_TOO_LARGE,
    CODE_UNSUPPORTED_MEDIA,
    CODE_UNSUPPORTED_SOURCE,
    CODE_INTERNAL,
    CODE_TTS_FAILED,
    CODE_TRANSCRIPTION_FAILED,
    CODE_LLM_SCHEMA_ERROR,
    CODE_DOWNLOAD_FAILED,
    CODE_JOB_NOT_READY,
    CODE_JOB_RUNNING,
    ErrorResponse,
    JobAccepted,
    LoomaResult,
    error_response,
)
from .pipeline.extract import LLMSchemaError
from .pipeline.ingest import (
    AudioTooLargeError,
    DownloadFailedError,
    InvalidURLError,
    PayloadTooLargeError,
    UnsupportedMediaError,
    UnsupportedSourceError,
)
from .storage.files import delete_job_files
from .storage.jobs import (
    JobStatus,
    get_default_db,
)
from .pipeline.narrate import TTSError
from .pipeline.orchestrator import JobSource, run_job, run_job_async
from .pipeline.transcribe import TranscriptionError
from .jobs import (
    JobManager,
    JobState,
    JobStatus as AsyncJobStatus,
    TERMINAL_STATUSES as ASYNC_TERMINAL_STATUSES,
    get_job_manager,
)

logger = logging.getLogger(__name__)


# --- AC-14 startup guards ---------------------------------------------------


def _check_ffmpeg_or_exit() -> None:
    """Exit with a clear error if ffmpeg/ffprobe is missing (AC-14)."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)]
    if missing:
        sys.stderr.write(
            "Looma startup aborted: ffmpeg/ffprobe not found on PATH "
            f"({', '.join(missing)}). Install with: "
            "`sudo apt-get install -y ffmpeg`.\n"
        )
        sys.exit(1)


def _check_llm_key_or_warn() -> None:
    """Warn if no LLM key is configured, but don't crash (AC-14 relax)."""
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        sys.stderr.write(
            "WARNING: no LLM API key configured. Set ANTHROPIC_API_KEY or "
            "OPENAI_API_KEY as an environment variable (or HF Space secret). "
            "LLM extraction will fail until one is set.\n"
        )


# Run guards at import time.
_check_ffmpeg_or_exit()
_check_llm_key_or_warn()


# --- App factory ------------------------------------------------------------


#: Path to the static frontend directory. The ``index.html`` at the
#: root of this directory is served at ``/`` (AC-8); sibling
#: resources (``styles.css``, ``app.js``) are served at their
#: relative paths.
_FRONTEND_DIR: Path = Path(
    os.environ.get(
        "FRONTEND_DIR",
        str(Path(__file__).resolve().parents[2] / "frontend"),
    )
).resolve()


def create_app() -> FastAPI:
    """Build and return the FastAPI app.

    Kept as a factory so tests can build isolated app instances
    (AC-13 / AC-11 will exercise this).
    """
    app = FastAPI(
        title="Looma",
        version="0.1.0",
        description=(
            "Convert YouTube links or uploaded videos into structured, "
            "reusable knowledge — refined title, summary, insights, "
            "chapter markers, and an audio-friendly narration."
        ),
    )

    # --- Dev-mode: disable caching of static files --------------------
    # The frontend is static HTML/JS/CSS served by FastAPI.  Without
    # an explicit ``Cache-Control`` header the browser and any
    # intermediate proxy (Cloudflare, API gateway, etc.) may cache an
    # old ``app.js``, making UI changes invisible to the end-user
    # until they manually bust the cache.  We set ``no-cache`` here
    # so every reload fetches the latest version.
    @app.middleware("http")
    async def _no_cache_static(request: Request, call_next: RequestResponseEndpoint) -> Response:  # type: ignore[misc]
        response = await call_next(request)
        path = request.url.path
        if path in ("/", "/app.js", "/styles.css") or path.startswith("/_"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    # --- AC-11: centralized exception handlers --------------------------
    #
    # Every non-2xx response goes through one of these so the body
    # is always ``{"error": <msg>, "code": <machine_code>}``. Without
    # them, FastAPI's default handlers emit ``{"detail": ...}`` for
    # unknown routes, request-validation errors, and 404s from the
    # static file mount, which would violate the AC-11 contract.
    #
    # Note: we register handlers for *both* ``HTTPException`` (the
    # FastAPI class) and ``StarletteHTTPException`` (its parent).
    # Starlette's ``StaticFiles`` mount raises the Starlette variant
    # when a file is missing, so catching only the FastAPI variant
    # would leave static-404s in the default ``{"detail": ...}``
    # shape.
    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Convert any HTTPException into the canonical AC-11 shape.

        Route handlers that raise :class:`HTTPException` directly (e.g.
        FastAPI's request-validation machinery) get the same
        ``{"error", "code"}`` body as the explicit ``JSONResponse``
        calls in :func:`api_extract` and the ``/api/jobs`` handlers.
        The status itself is normalized into the AC-11 allow-list
        first (e.g. 405 -> 404).
        """
        # ``exc.detail`` is conventionally a string, but Starlette
        # allows structured detail objects. Force to ``str`` so the
        # canonical contract (``error: str``) is preserved.
        message = str(exc.detail) if exc.detail is not None else "HTTP error."
        status = _normalize_status(exc.status_code)
        code = _machine_code_for_status(status)
        return _json_error(message, code, status)

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Convert Pydantic 422s into AC-11 400 INVALID_URL.

        AC-11 only enumerates 200/400/404/413/415/500; 422 isn't on
        the list, so we map validation errors down to 400 (the
        closest status the contract allows) with the canonical
        body. The first Pydantic error's ``loc`` + ``msg`` is
        flattened into a single human-readable line so the user
        sees *which* field was wrong.
        """
        message = _flatten_validation_error(exc)
        return _json_error(message, CODE_INVALID_URL, 400)

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Last-resort 500 for any exception that escapes the routes.

        Defense in depth: every route handler already catches its
        own stage-specific exceptions, but a stray ``RuntimeError``
        in a future endpoint would otherwise return a 500 with
        FastAPI's ``{"detail": "Internal Server Error"}`` body
        and a stack trace in the logs. With this handler the body
        stays canonical and the log line is structured.
        """
        logger.exception(
            "unhandled exception in %s %s: %s",
            request.method, request.url.path, exc,
        )
        return _json_error(
            f"Internal error: {exc}", CODE_INTERNAL, 500
        )

    # --- Routes (AC-7) --------------------------------------------------

    @app.get(
        "/audio/{job_id}.mp3",
        responses={
            200: {"content": {"audio/mpeg": {}}},
            404: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
        summary="Serve the TTS-generated narration MP3 for a job.",
    )
    def get_audio(job_id: str):
        """Return the TTS-generated MP3 at ``data/outputs/<job_id>.mp3``.

        Returns 404 with code ``NOT_FOUND`` if the file does not exist
        (so the frontend can fall back gracefully when a job has not
        finished yet). 500 errors use ``INTERNAL_ERROR``.

        The error body is the canonical ``{"error", "code"}`` shape
        (AC-11) — we return a :class:`JSONResponse` directly so the
        body is not wrapped in FastAPI's ``{"detail": ...}`` envelope.
        """
        # Defend against path traversal: ``job_id`` is used in a path
        # and could in principle contain ``..`` or a separator. We
        # reject anything that's not a safe identifier.
        if not _is_safe_job_id(job_id):
            return JSONResponse(
                status_code=400,
                content=error_response(
                    f"Invalid job_id: {job_id!r}", code=CODE_NOT_FOUND
                ),
            )

        path = OUTPUTS_DIR / f"{job_id}.mp3"
        if not path.exists() or not path.is_file():
            return JSONResponse(
                status_code=404,
                content=error_response(
                    f"No narration MP3 found for job_id={job_id!r}.",
                    code=CODE_NOT_FOUND,
                ),
            )

        # FileResponse handles range requests and ``Content-Length``
        # automatically, so the browser's ``<audio>`` element can
        # seek through the file without re-downloading.
        return FileResponse(
            path=str(path),
            media_type="audio/mpeg",
            filename=f"{job_id}.mp3",
        )

    # --- Routes (AC-8) --------------------------------------------------
    #
    # Both ``POST /api/extract`` and ``POST /api/extract/async`` route
    # to the same internal ``_submit_async_job`` helper (AC-1 + AC-3 +
    # AC-4). The legacy ``/api/extract`` URL used to be the synchronous
    # blocking handler; the 524 timeout that surfaced in production made
    # that contract a footgun, so both URLs now share the async handler.
    # The legacy URL is preserved verbatim so that any external script /
    # curl that targets it still gets a fast 202 (the same response body
    # as ``/api/extract/async``) instead of a 524.
    @app.post(
        "/api/extract",
        response_model=JobAccepted,
        status_code=202,
        responses={
            202: {"model": JobAccepted},
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
        },
        summary="Submit a job asynchronously. Returns 202 + job_id immediately.",
    )
    async def api_extract(request: Request) -> JSONResponse:
        """Async-submit alias of :func:`api_extract_async` (AC-1, AC-3).

        Historically this was the synchronous endpoint that ran the
        full pipeline and returned the :class:`LoomaResult` inline —
        fine on a fast loopback, but a Cloudflare 524 whenever a real
        source took more than the edge timeout. The contract is now
        identical to ``/api/extract/async``: the handler validates
        the body, allocates an in-memory job, spawns the pipeline in
        an ``asyncio`` background task, and returns ``202 + job_id``
        in milliseconds. The pipeline progress and final result are
        fetched via ``status_url`` and ``result_url`` respectively.
        """
        return await _submit_async_job(request)

    @app.post(
        "/api/extract/async",
        response_model=JobAccepted,
        status_code=202,
        responses={
            202: {"model": JobAccepted},
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
        },
        summary="Submit a job asynchronously. Returns 202 + job_id immediately.",
    )
    async def api_extract_async(request: Request) -> JSONResponse:
        """Accept a YouTube URL or file upload and return 202 immediately.

        The full pipeline (ingest → transcribe → extract → narrate)
        runs in a background ``asyncio`` task so the HTTP request
        itself returns in milliseconds. The client polls
        ``status_url`` (``GET /api/jobs/{id}``) for progress and
        ``result_url`` (``GET /api/jobs/{id}/result``) for the final
        :class:`LoomaResult`.

        The endpoint is the fix for the 524 timeout that the
        synchronous ``/api/extract`` exposes when Cloudflare (or any
        edge proxy) sits in front of the backend. With this endpoint
        the longest possible request is the small JSON
        ``JobAccepted`` 202, well under any proxy timeout.

        Both ``/api/extract`` and ``/api/extract/async`` route to the
        same internal :func:`_submit_async_job` helper (AC-4) so the
        legacy URL also returns 202 instead of holding the request
        open for the entire pipeline.

        Concurrency is capped by :data:`app.config.MAX_CONCURRENT_JOBS`
        via a semaphore; the background task waits its turn rather
        than OOM-ing the host.
        """
        return await _submit_async_job(request)

    # --- Routes (AC-9) --------------------------------------------------

    @app.get(
        "/api/jobs",
        responses={200: {"description": "Most recent jobs, newest first."}},
        summary="List the most recent jobs (AC-9).",
    )
    def list_jobs(
        limit: int = Query(
            default=DEFAULT_JOBS_LIMIT,
            ge=1,
            le=200,
            description="Maximum number of jobs to return (1-200, default 20).",
        ),
    ) -> JSONResponse:
        """Return the most recent ``limit`` jobs as a JSON array.

        Each entry has the seven AC-9 columns: ``id``,
        ``source_type``, ``source_ref``, ``title``, ``created_at``,
        ``duration_seconds``, ``status``.
        """
        jobs_db = get_default_db()
        records = jobs_db.list_jobs(limit=limit)
        return JSONResponse(
            status_code=200,
            content=jobs_db.to_dict_list(records),
        )

    @app.get(
        "/api/jobs/{job_id}",
        responses={
            200: {"description": "Job status (async) or DB row (legacy sync)."},
            404: {"model": ErrorResponse},
        },
        summary="Fetch a single job by id (AC-9 + async status).",
    )
    def get_job(job_id: str) -> JSONResponse:
        """Return the status of ``job_id``.

        Looks up the in-memory :class:`~app.jobs.JobManager` first
        (the new async ``/api/extract/async`` path) and falls back to
        the JSON-backed jobs file (the synchronous
        ``/api/extract`` path). 404 if neither has it.
        """
        if not _is_safe_job_id(job_id):
            return JSONResponse(
                status_code=400,
                content=error_response(
                    f"Invalid job_id: {job_id!r}", code=CODE_NOT_FOUND
                ),
            )

        # 1) Async jobs (in-memory, new path)
        async_job = get_job_manager().get(job_id)
        if async_job is not None:
            return JSONResponse(
                status_code=200,
                content=_async_job_to_status_dict(async_job),
            )

        # 2) Legacy sync jobs (JSON-backed file)
        jobs_db = get_default_db()
        record = jobs_db.get_job(job_id)
        if record is not None:
            return JSONResponse(
                status_code=200,
                content=jobs_db.to_dict_list([record])[0],
            )

        return JSONResponse(
            status_code=404,
            content=error_response(
                f"No job found for job_id={job_id!r}.",
                code=CODE_NOT_FOUND,
            ),
        )

    @app.get(
        "/api/jobs/{job_id}/result",
        responses={
            200: {"description": "The job's final LoomaResult."},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
        },
        summary="Fetch the final LoomaResult of an async job.",
    )
    async def get_async_job_result(job_id: str) -> JSONResponse:
        """Return the :class:`LoomaResult` of an async job, or 409 if not done.

        404 if the job_id is unknown. 409 ``JOB_NOT_READY`` if the
        job is still running, failed, or timed out. The body of a
        failed / timed-out job is a canonical
        ``{"error", "code"}`` response (not a 200 with a partial
        result) so the JS client can show the error message verbatim.
        """
        if not _is_safe_job_id(job_id):
            return JSONResponse(
                status_code=400,
                content=error_response(
                    f"Invalid job_id: {job_id!r}", code=CODE_NOT_FOUND
                ),
            )
        job = get_job_manager().get(job_id)
        if job is None:
            return JSONResponse(
                status_code=404,
                content=error_response(
                    f"No async job found for job_id={job_id!r}.",
                    code=CODE_NOT_FOUND,
                ),
            )
        if job.status == AsyncJobStatus.DONE:
            assert job.result is not None  # set by _run_async_job
            return JSONResponse(status_code=200, content=job.result)
        if job.status in (AsyncJobStatus.FAILED, AsyncJobStatus.TIMEOUT):
            err = job.error or {
                "code": "INTERNAL_ERROR",
                "msg": f"Job ended in status={job.status.value!r}.",
            }
            status = 500 if job.status == AsyncJobStatus.FAILED else 408
            return JSONResponse(
                status_code=status,
                content=error_response(err.get("msg", "Job failed."), err["code"]),
            )
        # Still running (queued / downloading / transcribing).
        return JSONResponse(
            status_code=409,
            content=error_response(
                f"Job {job_id!r} is not ready (status={job.status.value!r}).",
                code=CODE_JOB_NOT_READY,
            ),
        )

    @app.delete(
        "/api/jobs/{job_id}",
        responses={
            200: {"description": "Deletion report (row + files)."},
            404: {"model": ErrorResponse},
        },
        summary="Delete a job row and its on-disk MP3s (AC-10).",
    )
    def delete_job(job_id: str) -> JSONResponse:
        if not _is_safe_job_id(job_id):
            return JSONResponse(
                status_code=400,
                content=error_response(
                    f"Invalid job_id: {job_id!r}", code=CODE_NOT_FOUND
                ),
            )
        jobs_db = get_default_db()
        record = jobs_db.get_job(job_id)
        if record is None:
            return JSONResponse(
                status_code=404,
                content=error_response(
                    f"No job found for job_id={job_id!r}.",
                    code=CODE_NOT_FOUND,
                ),
            )
        removed = jobs_db.delete_job(job_id)
        files = delete_job_files(job_id)
        return JSONResponse(
            status_code=200,
            content={
                "deleted": bool(removed),
                "job_id": job_id,
                "files_removed": files,
            },
        )

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    # --- Static frontend (AC-8) ---------------------------------------
    # The frontend is a single-page app: ``index.html`` at the root,
    # with ``styles.css`` and ``app.js`` as siblings. We mount the
    # ``StaticFiles`` handler at ``/`` so ``GET /`` returns the
    # ``index.html`` and ``GET /styles.css`` / ``GET /app.js`` return
    # the corresponding assets. The HTML's relative paths then
    # resolve naturally.
    if _FRONTEND_DIR.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(_FRONTEND_DIR), html=True),
            name="frontend",
        )
    else:  # pragma: no cover - defensive
        logger.warning(
            "Frontend directory %s does not exist; "
            "GET / will return 404.", _FRONTEND_DIR,
        )

    # --- Background tasks: sweeper + watchdog -----------------------
    #
    # Two long-running ``asyncio`` tasks are scheduled on the event
    # loop to maintain the in-memory :class:`~app.jobs.JobManager`:
    #
    # * ``_sweeper_loop`` evicts entries older than
    #   ``JOB_TTL_SECONDS`` every ``JOB_SWEEP_INTERVAL_SECONDS``.
    # * ``_watchdog_loop`` flips any in-flight job past its
    #   ``deadline`` to ``status=timeout`` every
    #   ``JOB_WATCHDOG_INTERVAL_SECONDS``.
    #
    # The two tasks are tracked on ``app.state`` so tests can cancel
    # them deterministically via ``app.dependency_overrides`` or by
    # waiting on the event loop to drain at teardown.

    @app.on_event("startup")
    async def _start_background_loops() -> None:
        app.state._sweeper_task = asyncio.create_task(
            _sweeper_loop(),
            name="looma-sweeper",
        )
        app.state._watchdog_task = asyncio.create_task(
            _watchdog_loop(),
            name="looma-watchdog",
        )
        logger.info(
            "Started background loops: sweeper every %ds, "
            "watchdog every %ds, TTL=%ds",
            JOB_SWEEP_INTERVAL_SECONDS,
            JOB_WATCHDOG_INTERVAL_SECONDS,
            JOB_TTL_SECONDS,
        )

    @app.on_event("shutdown")
    async def _stop_background_loops() -> None:
        for attr in ("_sweeper_task", "_watchdog_task"):
            t = getattr(app.state, attr, None)
            if t is not None and not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        logger.info("Stopped background loops")

    return app


# --- Module-level helpers (used by the async endpoints above) --------------


async def _submit_async_job(request: Request) -> JSONResponse:
    """Submit a job asynchronously and return ``202 + job_id``.

    Single source of truth for the submit route (AC-4). Both
    ``POST /api/extract`` and ``POST /api/extract/async`` delegate to
    this helper so the request handling — body parsing, validation,
    in-memory job allocation, background-task spawn, 202 response —
    lives in exactly one place.

    Returns:
        A :class:`JSONResponse` with status 202 and the canonical
        :class:`~app.models.JobAccepted` body when the submission is
        accepted. Returns a 4xx :class:`JSONResponse` immediately if
        the body is malformed, has no input, or carries both a
        ``youtube_url`` and a file upload.

    The handler never awaits the pipeline. The heavy work runs in an
    ``asyncio.create_task`` background coroutine (:func:`_run_async_job`)
    so the wall-clock time of the request is bounded by the body
    parse + DB write + task spawn — typically a few milliseconds, well
    under the 500 ms AC-1 budget.
    """
    content_type = (request.headers.get("content-type") or "").lower()
    youtube_url: str | None = None
    upload_path: Path | None = None
    upload_filename: str | None = None

    # --- 1) Pre-flight: parse the body ------------------------------

    if content_type.startswith("application/json"):
        try:
            payload = await request.json()
        except Exception as exc:
            return _json_error(
                f"Invalid JSON body: {exc}", CODE_INVALID_URL, 400
            )
        if not isinstance(payload, dict):
            return _json_error(
                "JSON body must be an object.", CODE_INVALID_URL, 400
            )
        err = _validate_youtube_url_field(payload.get("youtube_url"))
        if err is not None:
            return err
        youtube_url = payload.get("youtube_url")  # type: ignore[assignment]

    elif content_type.startswith("multipart/form-data"):
        form = await request.form()
        url_value = form.get("youtube_url")
        err = _validate_youtube_url_field(url_value)
        if err is not None:
            return err
        youtube_url = url_value  # type: ignore[assignment]
        upload = form.get("file")
        if upload is not None and hasattr(upload, "filename"):
            # ``upload`` is a starlette ``UploadFile`` when parsed
            # from a multipart form. Spool the bytes to a temp file
            # so the background task can ``Path(...)``-read it the
            # same way it reads a YouTube download.
            upload_filename = getattr(upload, "filename", None) or "upload"
            suffix = Path(upload_filename).suffix or ".mp4"
            fd, name = tempfile.mkstemp(suffix=suffix, prefix="upload-")
            upload_path = Path(name)
            with os.fdopen(fd, "wb") as f:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
    else:
        return _json_error(
            f"Unsupported Content-Type: {content_type!r}. "
            "Use application/json or multipart/form-data.",
            CODE_INVALID_URL, 400,
        )

    # --- 2) Pre-flight: shape validation ----------------------------
    if youtube_url is not None and upload_path is not None:
        return _json_error(
            "Provide either youtube_url or file, not both.",
            CODE_INVALID_URL, 400,
        )
    if youtube_url is None and upload_path is None:
        return _json_error(
            "Missing input: provide youtube_url or a file upload.",
            CODE_INVALID_URL, 400,
        )

    # --- 3) Dedup: reuse existing job for same YouTube video ---------
    # If the user submits the same YouTube URL twice, return the
    # existing job ID so they don't have to wait for a duplicate
    # pipeline.  File uploads can't be deduped without a content hash.
    source_ref = (
        upload_filename if upload_path is not None else (youtube_url or "")
    )
    video_id: str | None = None
    manager = get_job_manager()
    if youtube_url is not None:
        video_id = _extract_youtube_video_id(youtube_url)
        if video_id is not None:
            existing = manager.find_by_source_ref(video_id)
            if existing is not None:
                body = JobAccepted(
                    job_id=existing.id,
                    status=existing.status.value,
                    status_url=f"/api/jobs/{existing.id}",
                    result_url=f"/api/jobs/{existing.id}/result",
                )
                logger.info(
                    "dedup: job %s already running for video %s; "
                    "returning existing job",
                    existing.id, video_id,
                )
                return JSONResponse(
                    status_code=202,
                    content=body.model_dump(mode="json"),
                )

    # --- 4) Allocate the in-memory job ------------------------------
    kind = "upload" if upload_path is not None else "youtube"
    # Use the video ID as the stable ``source_ref`` for dedup, not
    # the full URL (which may differ by tracking parameters, capitalisation, etc.)
    if youtube_url is not None and video_id is not None:
        source_ref = video_id
    job = manager.create(kind=kind, source_ref=source_ref)

    # --- 4) Spawn the background pipeline task ---------------------
    # The handler does not await this task — the request returns as
    # soon as the task is scheduled. The watchdog will cancel it if
    # it runs past the deadline, the sweeper evicts the state once
    # it reaches a terminal status, and the task self-cleans its
    # own temp files in the finally block.
    task = asyncio.create_task(
        _run_async_job(
            job_id=job.id,
            youtube_url=youtube_url,
            upload_path=upload_path,
            upload_filename=upload_filename,
        ),
        name=f"looma-job-{job.id}",
    )
    manager.attach_task(job.id, task)
    # Detach when the task finishes so the table doesn't grow
    # forever with completed tasks. Done-callback is fire-and-forget.
    task.add_done_callback(lambda _t, jid=job.id: _on_async_job_done(jid, _t))

    # --- 5) Return 202 + JobAccepted --------------------------------
    body = JobAccepted(
        job_id=job.id,
        status="queued",
        status_url=f"/api/jobs/{job.id}",
        result_url=f"/api/jobs/{job.id}/result",
    )
    return JSONResponse(
        status_code=202,
        content=body.model_dump(mode="json"),
    )


async def _sweeper_loop() -> None:
    """Periodically evict expired entries from the JobManager.

    Runs forever (until cancelled at app shutdown). Sleeps
    :data:`app.config.JOB_SWEEP_INTERVAL_SECONDS` between sweeps.
    """
    while True:
        try:
            await asyncio.sleep(JOB_SWEEP_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            return
        try:
            removed = get_job_manager().sweep_expired(JOB_TTL_SECONDS)
            if removed:
                logger.info("sweeper: removed %d expired jobs", removed)
        except Exception as exc:  # noqa: BLE001
            logger.exception("sweeper loop error: %s", exc)


async def _watchdog_loop() -> None:
    """Periodically flip jobs past their deadline to ``status=timeout``.

    The deadline is set at job creation (default
    :data:`app.config.JOB_TIMEOUT_SECONDS` = 2 hours). A stuck
    pipeline is the failure mode this guards against: the original
    HTTP request that spawned it has long since returned, so the
    event loop would otherwise happily run the pipeline until
    Whisper itself decides to crash.
    """
    while True:
        try:
            await asyncio.sleep(JOB_WATCHDOG_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            return
        try:
            flipped = get_job_manager().sweep_timeouts()
            for jid in flipped:
                logger.warning("watchdog: job %s timed out", jid)
        except Exception as exc:  # noqa: BLE001
            logger.exception("watchdog loop error: %s", exc)


async def _run_async_job(
    job_id: str,
    *,
    youtube_url: str | None,
    upload_path: Path | None,
    upload_filename: str | None,
) -> None:
    """Background coroutine: run the full pipeline for one async job.

    Updates the in-memory :class:`JobState` at three points:

    * ``status=downloading`` just before the ingest stage starts
      (covers both the YouTube ``yt-dlp`` download and the
      ffmpeg-from-upload conversion).
    * ``status=transcribing`` just before the Whisper stage starts
      (the longest single stage on a CPU-only host).
    * ``status=done`` with the serialized :class:`LoomaResult` once
      the pipeline returns, or ``status=failed`` with an
      ``error: {code, msg}`` on any stage-specific exception.

    The semaphore from :meth:`JobManager.get_semaphore` is awaited
    first so a single host can't accept more parallel pipelines
    than :data:`app.config.MAX_CONCURRENT_JOBS`.
    """
    manager = get_job_manager()
    semaphore = manager.get_semaphore()

    try:
        async with semaphore:
            # --- Build the JobSource --------------------------------
            if upload_path is not None:
                source = JobSource.upload(
                    str(upload_path), display_name=upload_filename
                )
            else:
                source = JobSource.youtube(youtube_url or "")

            # --- Stage 1: download / convert -----------------------
            manager.update(
                job_id,
                status=AsyncJobStatus.DOWNLOADING,
                stage_msg="Downloading audio",
                progress=5,
            )

            # Map orchestrator stage names → the three visible
            # UI states. The orchestrator emits ``ingest``,
            # ``transcribe``, ``extract``, ``narrate`` in order;
            # we fold ``extract`` and ``narrate`` into
            # ``transcribing`` for the UI (the LLM call and the
            # TTS synthesis are both sub-second on a healthy
            # backend, so a separate stage indicator would just
            # flicker past the user).
            def _on_stage(stage: str) -> None:
                if stage == "ingest":
                    manager.update(
                        job_id,
                        status=AsyncJobStatus.DOWNLOADING,
                        stage_msg="Downloading audio",
                        progress=10,
                    )
                elif stage in ("transcribe", "extract", "narrate"):
                    label = {
                        "transcribe": "Transcribing audio",
                        "extract": "Extracting knowledge",
                        "narrate": "Generating narration",
                    }[stage]
                    manager.update(
                        job_id,
                        status=AsyncJobStatus.TRANSCRIBING,
                        stage_msg=label,
                        progress=60 if stage == "transcribe" else 85,
                    )

            # Granular transcribe progress: the Whisper chunking
            # callback fires a 0..100 percentage that we remap to
            # the 60-85 range of the overall job progress, so the
            # frontend bar moves smoothly during the long
            # CPU-bound transcription step instead of freezing at
            # 60 % until the stage finishes.
            def _on_transcribe_progress(pct: int) -> None:
                overall = 60 + int(pct * 0.25)
                manager.update(
                    job_id,
                    status=AsyncJobStatus.TRANSCRIBING,
                    stage_msg=f"Transcribing audio ({pct}%)",
                    progress=overall,
                )

            # ``run_job_async`` wraps the synchronous pipeline in
            # ``asyncio.to_thread`` so we don't block the event loop
            # while Whisper is loading / transcribing / narrating.
            result = await run_job_async(
                job_id=job_id, source=source, progress_callback=_on_stage,
                transcribe_progress_callback=_on_transcribe_progress,
            )

            # --- Finalize ------------------------------------------
            manager.update(
                job_id,
                status=AsyncJobStatus.DONE,
                stage_msg="Done",
                progress=100,
                result=result.model_dump(mode="json"),
            )
            logger.info("async job %s done", job_id)

    except asyncio.CancelledError:
        # The watchdog cancelled us because we ran past the deadline.
        # The watchdog has already flipped status to ``timeout``;
        # just log and let the finally block clean up the temp file.
        logger.warning("async job %s cancelled (timeout?)", job_id)
        raise
    except _PIPELINE_EXCEPTIONS as exc:
        # Stage-specific failure — keep the same shape as the
        # synchronous endpoint so the JS client can render the
        # same error UI.
        status, code = _STAGE_ERROR_MAP[type(exc)]
        manager.update(
            job_id,
            status=AsyncJobStatus.FAILED,
            stage_msg=exc.message,
            error={"code": code, "msg": exc.message},
        )
        logger.warning("async job %s failed: %s", job_id, exc.message)
    except Exception as exc:  # noqa: BLE001 - last-resort
        manager.update(
            job_id,
            status=AsyncJobStatus.FAILED,
            stage_msg="Internal error",
            error={"code": CODE_INTERNAL, "msg": f"Internal error: {exc}"},
        )
        logger.exception("async job %s failed unexpectedly: %s", job_id, exc)
    finally:
        # Belt-and-braces cleanup if the ingest stage didn't already
        # unlink the temp file.
        if upload_path is not None and upload_path.exists():
            try:
                upload_path.unlink()
            except OSError:  # pragma: no cover
                pass


def _on_async_job_done(job_id: str, task: "asyncio.Task") -> None:
    """Drop the ``asyncio.Task`` from the manager's bookkeeping.

    Called as a ``done_callback`` on the background task. The job
    state itself stays in the table until the sweeper evicts it
    (after ``JOB_TTL_SECONDS``); only the task reference is
    forgotten so the manager doesn't grow forever.
    """
    try:
        get_job_manager().detach_task(job_id)
    except Exception:  # noqa: BLE001 - defensive
        pass


def _async_job_to_status_dict(job: JobState) -> dict:
    """Serialize an in-memory :class:`JobState` to a status response body.

    Shape matches what the JS polling loop expects:

    * ``status`` — the lifecycle state (``queued`` / ``downloading``
      / ``transcribing`` / ``done`` / ``failed`` / ``timeout``).
    * ``progress`` — 0..100 integer for the progress bar.
    * ``stage_msg`` — human-readable label for the current step.
    * ``error`` — ``{code, msg}`` for ``failed`` / ``timeout`` jobs,
      absent otherwise.

    Plus the AC-9 fields (``id``, ``source_type``, ``source_ref``,
    ``created_at``) so the legacy ``/api/jobs`` listing still works
    against async jobs.
    """
    body: dict = {
        "id": job.id,
        "job_id": job.id,
        "source_type": job.kind,
        "source_ref": job.source_ref,
        "status": job.status.value,
        "progress": job.progress,
        "stage_msg": job.stage_msg,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
    if job.error is not None:
        body["error"] = job.error
    return body


def _json_error(message: str, code: str, status: int) -> JSONResponse:
    """Build a :class:`JSONResponse` with the canonical error shape.

    Tiny wrapper so the route handler stays small. The body is
    ``{"error": <message>, "code": <machine_code>}`` (AC-11).
    """
    return JSONResponse(
        status_code=status,
        content=error_response(message, code=code),
    )


def _validate_youtube_url_field(value: object) -> JSONResponse | None:
    """Validate that ``value`` is ``None`` or a string.

    Both ``application/json`` and ``multipart/form-data`` carry
    a ``youtube_url`` field. We reject any non-string value with
    the canonical 400 INVALID_URL response so a JS client that
    sent a number / object / array gets the same error shape as
    every other 4xx.

    Returns ``None`` when ``value`` is acceptable (``None`` or
    ``str``), or a :class:`JSONResponse` to short-circuit the
    request.
    """
    if value is None or isinstance(value, str):
        return None
    return _json_error(
        "youtube_url must be a string.", CODE_INVALID_URL, 400
    )


# --- AC-11: HTTP status -> machine code mapping ----------------------------
#
# AC-11 requires every error response to use the canonical
# ``{"error", "code"}`` body and a proper HTTP status code from
# the fixed set {400, 404, 413, 415, 500}. Status code 200 is the
# success path and is not mapped here.
#
# This table is the single source of truth for "what machine-readable
# code goes with what HTTP status when we don't have a more specific
# exception to inspect". Stage-specific exceptions in the route
# handlers pick a more specific code (INVALID_URL, PAYLOAD_TOO_LARGE,
# etc.) and bypass this table.
#
# We also normalize the status itself: 405 (Method Not Allowed) is
# not in the AC-11 allow-list, so it is mapped down to 404 with the
# code NOT_FOUND. Semantically a 405 says "this resource does not
# support this method", which is what 404 + NOT_FOUND conveys to a
# JS client that doesn't read the status code.
_STATUS_TO_CODE: dict[int, str] = {
    400: CODE_INVALID_URL,        # generic "bad request" -> INVALID_URL
    404: CODE_NOT_FOUND,          # generic "not found"
    405: CODE_NOT_FOUND,          # method not allowed -> 404 NOT_FOUND
    413: CODE_PAYLOAD_TOO_LARGE,  # generic "too large"
    415: CODE_UNSUPPORTED_MEDIA,  # generic "unsupported media"
    500: CODE_INTERNAL,           # generic "internal error"
}

#: HTTP statuses that are remapped into the AC-11 allow-list.
#: 405 is the only status we currently normalize (down to 404).
_STATUS_REMAP: dict[int, int] = {
    405: 404,
}


def _normalize_status(status: int) -> int:
    """Remap ``status`` into the AC-11 allow-list (200/400/404/413/415/500).

    Starlette raises ``HTTPException(405)`` for method-not-allowed;
    that status isn't in the AC-11 contract, so we collapse it
    to 404. Anything we don't know about passes through
    unchanged — the :func:`_machine_code_for_status` lookup will
    still emit the canonical body, but a future maintainer can
    decide whether the original status warrants its own code.
    """
    return _STATUS_REMAP.get(status, status)


def _machine_code_for_status(status: int) -> str:
    """Return the canonical machine code for ``status`` (AC-11).

    Falls back to :data:`CODE_INTERNAL` for any unmapped status so
    the response is always shape-stable, even if a new HTTPException
    status sneaks in via a future FastAPI version.
    """
    return _STATUS_TO_CODE.get(status, CODE_INTERNAL)


# --- AC-11: pipeline-stage exception -> (status, code) ---------------------
#
# Every stage in the pipeline raises a stage-specific exception
# (InvalidURLError, TranscriptionError, TTSError, ...). The API
# layer maps each one to an HTTP status + machine code. Encoding
# the mapping as a tuple-by-class table (rather than 9 separate
# ``except`` blocks) keeps the contract auditable in one place and
# makes it obvious that there is no overlap between codes.
_STAGE_ERROR_MAP: dict[type[Exception], tuple[int, str]] = {
    # 4xx — client errors
    InvalidURLError:         (400, CODE_INVALID_URL),
    UnsupportedSourceError:  (400, CODE_UNSUPPORTED_SOURCE),
    UnsupportedMediaError:   (415, CODE_UNSUPPORTED_MEDIA),
    PayloadTooLargeError:    (413, CODE_PAYLOAD_TOO_LARGE),
    AudioTooLargeError:      (413, CODE_PAYLOAD_TOO_LARGE),
    # 5xx — server / pipeline errors
    DownloadFailedError:      (500, CODE_DOWNLOAD_FAILED),
    TranscriptionError:      (500, CODE_TRANSCRIPTION_FAILED),
    LLMSchemaError:          (500, CODE_LLM_SCHEMA_ERROR),
    TTSError:                (500, CODE_TTS_FAILED),
}
#: Tuple of all stage exceptions the API layer knows how to map.
#: Used as the ``except`` clause so a new stage just needs to be
#: added to :data:`_STAGE_ERROR_MAP`.
_PIPELINE_EXCEPTIONS: tuple[type[Exception], ...] = tuple(_STAGE_ERROR_MAP)


def _mark_job_failed(jobs_db, job_id: str, exc: BaseException) -> None:
    """Mark a job as failed in the JSON store. Best-effort: errors are swallowed.

    A failure here is logged but never raised — the user-facing
    error response has already been decided by the caller, and a
    broken DB write must not flip the HTTP code from 4xx/5xx to
    500 just because the bookkeeping didn't take.
    """
    try:
        jobs_db.update_job(job_id, status=JobStatus.FAILED)
    except Exception:  # pragma: no cover - defensive
        logger.warning("update_job (failed) for %s failed: %s", job_id, exc)


def _flatten_validation_error(exc: RequestValidationError) -> str:
    """Build a short, human-readable summary of a validation error.

    FastAPI's default 422 body is a list of typed error objects; we
    flatten the first one into a single line because the AC-11
    contract is a single ``error`` string, not a structured array.
    """
    errors = exc.errors() or []
    if not errors:
        return "Invalid request."
    first = errors[0]
    loc = ".".join(str(part) for part in first.get("loc", []))
    msg = first.get("msg", "Invalid value.")
    return f"{loc}: {msg}" if loc else msg


# A small allow-list for ``job_id`` in the URL. Anything outside this
# set is rejected with 400 to keep the route immune to path traversal.
_JOB_ID_SAFE = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-_"
)


#: Regex to extract YouTube video IDs from common URL formats:
#: ``youtube.com/watch?v=VIDEO_ID``, ``youtu.be/VIDEO_ID``,
#: ``youtube-nocookie.com``, and shorts/embed variants.
_YT_VIDEO_ID_RE = re.compile(r"(?:v=|be/|shorts/|embed/)([a-zA-Z0-9_-]{11})")


def _extract_youtube_video_id(url: str) -> str | None:
    """Return the YouTube video ID from ``url``, or ``None`` if not found.

    Handles ``youtube.com/watch?v=...``, ``youtu.be/...``,
    ``youtube-nocookie.com/embed/...``, and ``/shorts/...`` URLs.
    """
    m = _YT_VIDEO_ID_RE.search(url)
    return m.group(1) if m else None


def _is_safe_job_id(job_id: str) -> bool:
    """True if ``job_id`` only contains characters that are safe in a URL path."""
    if not job_id or len(job_id) > 128:
        return False
    return all(c in _JOB_ID_SAFE for c in job_id)


# Module-level app for `uvicorn app.main:app`. The factory above is
# the canonical way to get an app instance.
app = create_app()


# --- Module exports ---------------------------------------------------------

#: Public surface of :mod:`app.main`. ``app`` and ``create_app``
#: are the canonical entry points; ``uvicorn`` resolves ``app.main:app``.
__all__ = ["app", "create_app"]
