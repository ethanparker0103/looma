"""Shared Pydantic schemas and error response shape.

All error responses from the API use shape:

    {"error": "<message>", "code": "<MACHINE_CODE>"}

as required by AC-11. The Pydantic models in this module are the single
source of truth for that contract; the FastAPI exception handlers
(in `app/main.py`, AC-11) construct responses from these classes.

Successful-response models also live here so that every pipeline stage
has one canonical type to return and the orchestrator can compose
them into a :class:`LoomaResult` (AC-6).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error body for every non-2xx response (AC-11)."""

    error: str = Field(..., description="Human-readable error message.")
    code: str = Field(..., description="Machine-readable error code.")


# --- Transcription (AC-4) ---------------------------------------------------


class TranscriptSegment(BaseModel):
    """One contiguous chunk of the transcript with start/end timestamps."""

    start: float = Field(..., ge=0.0, description="Segment start in seconds.")
    end: float = Field(..., ge=0.0, description="Segment end in seconds.")
    text: str = Field(..., description="Verbatim text of the segment.")


class TranscriptionResult(BaseModel):
    """Whisper transcription of the normalized MP3 (AC-4)."""

    transcript: str = Field(..., description="Full transcript as a single string.")
    segments: list[TranscriptSegment] = Field(
        default_factory=list,
        description="Time-stamped segments, ordered by start time.",
    )
    language: str = Field(..., description="Whisper-detected language code (e.g. 'en').")
    duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total audio duration in seconds, derived from the last segment end.",
    )


# --- LLM extraction (AC-5) -------------------------------------------------


class Chapter(BaseModel):
    """A chapter with start/end timestamps aligned to a transcript segment.

    AC-5 requires ``chapters`` to cover the full ``[0, duration]`` range
    with non-overlapping, contiguous slices after snapping.
    """

    start_seconds: float = Field(..., ge=0.0, description="Chapter start in seconds.")
    end_seconds: float = Field(..., ge=0.0, description="Chapter end in seconds.")
    title: str = Field(..., min_length=1, description="Short chapter title.")


class KnowledgeExtract(BaseModel):
    """Strict JSON contract returned by the LLM extractor (AC-5).

    The LLM is prompted to return exactly this shape; we re-validate the
    parsed JSON against this model and retry once on validation failure.
    Final failure surfaces as HTTP 500 with code ``LLM_SCHEMA_ERROR``.
    """

    title: str = Field(
        ..., max_length=120, description="Refined video title, 120 chars max."
    )
    summary: str = Field(
        ..., min_length=1, description="3-5 sentence executive summary."
    )
    insights: list[str] = Field(
        ..., min_length=5, max_length=10,
        description="5-10 imperative-form key insights as bullet strings.",
    )
    chapters: list[Chapter] = Field(
        ..., min_length=1, description="Chapters covering [0, duration]."
    )
    narrative: str = Field(
        ..., min_length=1,
        description="Audio-friendly narration, 150-400 words, filler-free.",
    )
    filler_removed: int = Field(
        ..., ge=0, description="Number of filler words/phrases removed."
    )


# --- Orchestrator result (AC-6) --------------------------------------------


class StageTiming(BaseModel):
    """Wall-clock time spent in one pipeline stage (AC-10 observability).

    Captured by the orchestrator with :func:`time.perf_counter` so it is
    unaffected by wall-clock adjustments (NTP, DST, etc.). ``seconds`` is
    rounded to milliseconds when the orchestrator emits it so the
    surface area is human-readable; sub-millisecond precision is not
    useful for a 5-minute budget.
    """

    stage: str = Field(..., description="Pipeline stage name, e.g. 'ingest'.")
    seconds: float = Field(..., ge=0.0, description="Wall-clock duration in seconds.")


class PipelineTimings(BaseModel):
    """Per-stage timings captured by :func:`app.pipeline.orchestrator.run_job`.

    The four stage durations sum to within rounding of ``total_seconds``.
    The orchestrator also logs these so a real run can be inspected
    from the log stream when AC-10's 5-minute budget is exceeded.
    """

    ingest_seconds: float = Field(..., ge=0.0, description="yt-dlp / ffmpeg-convert time.")
    transcribe_seconds: float = Field(..., ge=0.0, description="Whisper transcription time.")
    extract_seconds: float = Field(..., ge=0.0, description="LLM extract time.")
    narrate_seconds: float = Field(..., ge=0.0, description="TTS synthesis time.")
    total_seconds: float = Field(..., ge=0.0, description="Sum of the four stages.")


class LoomaResult(BaseModel):
    """The single response object the API returns for a finished job (AC-6).

    Composed by :func:`app.pipeline.orchestrator.run_job` from the outputs
    of the four pipeline stages (AC-2/3 ingest, AC-4 transcribe, AC-5
    extract, AC-7 narrate). The shape is the public API contract: every
    field below is what the frontend at ``/`` consumes (AC-8).

    Fields:
        job_id: The unique identifier for this job. Same value the
            caller used to submit the request.
        source_type: ``"youtube"`` or ``"upload"`` — recorded so the
            frontend can render the right input-tab state on revisit.
        source_ref: The original URL (for YouTube) or the basename of
            the upload (for uploads). Display-only.
        title: The LLM-refined title (AC-5). The frontend shows this
            prominently above the result sections.
        transcription: Full AC-4 output — verbatim text + segments
            + language + duration. The orchestrator passes it through
            unmodified.
        knowledge: Full AC-5 output — structured knowledge (title,
            summary, insights, chapters, narrative, filler_removed).
        audio_url: The public URL of the TTS-generated narration MP3
            (served at ``GET /audio/{job_id}.mp3`` per AC-7). Path
            form (``/audio/<job_id>.mp3``) so the frontend can bind
            it to an ``<audio>`` element directly.
        created_at: ISO-8601 UTC timestamp marking when the job
            finished. Suffix ``"Z"`` is the explicit UTC marker.
        timings: Per-stage wall-clock durations captured by the
            orchestrator (AC-10). Not rendered by the frontend
            (AC-8) but exposed on the API surface for observability
            and the AC-10 perf-budget tests.
    """

    job_id: str = Field(..., min_length=1, description="Unique job identifier.")
    source_type: str = Field(
        ..., description="'youtube' or 'upload'."
    )
    source_ref: str = Field(
        ..., description="Original URL or upload filename."
    )
    title: str = Field(
        ..., max_length=120, description="Refined video title from AC-5."
    )
    transcription: TranscriptionResult = Field(
        ..., description="Full AC-4 output."
    )
    knowledge: KnowledgeExtract = Field(
        ..., description="Full AC-5 output."
    )
    audio_url: str = Field(
        "", description="Public URL of the narrated MP3, e.g. /audio/<job_id>.mp3. "
        "Empty when TTS failed to generate audio."
    )
    created_at: str = Field(
        ..., description="ISO-8601 UTC timestamp (suffix 'Z')."
    )
    timings: PipelineTimings | None = Field(
        default=None,
        description=(
            "Per-stage wall-clock timings (AC-10). "
            "Optional for backward compatibility with older clients."
        ),
    )


# --- Error code constants ---------------------------------------------------
# These strings are part of the public API contract — do not rename without
# bumping the API version.

CODE_INVALID_URL = "INVALID_URL"
CODE_UNSUPPORTED_SOURCE = "UNSUPPORTED_SOURCE"
CODE_PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
CODE_UNSUPPORTED_MEDIA = "UNSUPPORTED_MEDIA"
CODE_VIDEO_TOO_LONG = "VIDEO_TOO_LONG"
CODE_NOT_FOUND = "NOT_FOUND"
CODE_LLM_SCHEMA_ERROR = "LLM_SCHEMA_ERROR"
CODE_TTS_FAILED = "TTS_FAILED"
CODE_TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
CODE_DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
CODE_INTERNAL = "INTERNAL_ERROR"
CODE_JOB_RUNNING = "JOB_RUNNING"  # 409 conflict when client polls for result too early
CODE_JOB_NOT_READY = "JOB_NOT_READY"  # 202 case (kept for parity; not used yet)


class JobAccepted(BaseModel):
    """Response body for the async ``POST /api/extract/async`` endpoint.

    The handler returns 202 Accepted with this body so the client
    can poll ``status_url`` for progress and ``result_url`` once the
    pipeline completes. Both URLs are absolute paths under the
    FastAPI app so a JS client on a different origin can append
    them to its base URL without server-side rewrites.
    """

    job_id: str = Field(..., description="UUID hex job identifier.")
    status: str = Field(
        default="running", description="Always 'running' at submission."
    )
    status_url: str = Field(
        ...,
        description="Path to poll for current status (GET /api/jobs/{id}).",
    )
    result_url: str = Field(
        ...,
        description=(
            "Path to fetch the final LoomaResult from "
            "(GET /api/jobs/{id}/result)."
        ),
    )


class CodeExtractResponse(BaseModel):
    """Response body for a completed extraction job (BYOK).

    The backend returns the raw transcription; the frontend is
    responsible for calling the LLM with the user's own API key
    and rendering the extracted knowledge.
    """

    transcription: dict[str, Any] = Field(
        ..., description="Full transcription result as a dict."
    )
    segments: list[dict[str, Any]] = Field(
        ..., description="Time-stamped segments [{start, end, text}]."
    )
    language: str = Field(
        default="en", description="Detected language code."
    )
    duration_seconds: float = Field(
        ..., description="Audio duration in seconds."
    )


def error_response(error: str, code: str) -> dict[str, Any]:
    """Build a dict that matches ErrorResponse.

    Used by exception handlers; the Pydantic model is the source of truth,
    so this helper simply constructs the underlying shape.
    """
    return ErrorResponse(error=error, code=code).model_dump()
