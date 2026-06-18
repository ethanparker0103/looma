"""Pipeline orchestrator — ingest + transcribe only (BYOK).

Public entry points:

* :class:`JobSource` — tagged-union for the two input shapes
  (YouTube URL, uploaded video file).
* :func:`run_job` — runs the two server-side stages and returns
  a :class:`~app.models.TranscriptionResult`.
* :func:`run_job_async` — async wrapper for :func:`run_job`.

Pipeline
--------
For every job::

    [ingest]        ->  data/audio/<job_id>.mp3     (AC-2 or AC-3)
    [transcribe]    ->  TranscriptionResult          (AC-4)

LLM extraction and TTS narration are handled entirely on the frontend
(BYOK — bring your own key). The backend only does CPU-bound work
(download + Whisper) that can't run in the browser.

Design notes
------------
* The orchestrator is synchronous. The FastAPI layer wraps
  :func:`run_job` in ``asyncio.to_thread`` so the event loop stays
  responsive while Whisper runs.
* :func:`_stage_*` helpers are extracted so tests can patch
  individual stages with ``mock.patch.object``.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Literal, TypeVar

from ..config import (
    AUDIO_DIR,
    MAX_PIPELINE_SECONDS,
    OUTPUTS_DIR,
    STAGE_BUDGET_INGEST_SECONDS,
    STAGE_BUDGET_TRANSCRIBE_SECONDS,
)
from ..models import TranscriptionResult
from . import ingest as ingest_mod
from . import transcribe as transcribe_mod

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


# --- JobSource --------------------------------------------------------------


class JobSource:
    """A validated input source for a pipeline job.

    Use the class-method constructors :meth:`youtube` and
    :meth:`upload` rather than instantiating directly.

    For uploads, ``display_name`` is the user-facing filename
    (e.g. ``"my-video.mp4"``) used for reference.
    """

    __slots__ = ("kind", "ref", "display_name")

    kind: Literal["youtube", "upload"]
    ref: str
    display_name: str | None

    def __init__(
        self,
        kind: str,
        ref: str,
        display_name: str | None = None,
    ) -> None:
        if kind not in ("youtube", "upload"):
            raise ValueError(
                f"JobSource kind must be 'youtube' or 'upload', got {kind!r}"
            )
        if not isinstance(ref, str) or not ref:
            raise ValueError("JobSource ref must be a non-empty string.")
        self.kind = kind
        self.ref = ref
        self.display_name = display_name

    @classmethod
    def youtube(cls, url: str) -> "JobSource":
        return cls("youtube", url)

    @classmethod
    def upload(
        cls, path: str | Path, display_name: str | None = None
    ) -> "JobSource":
        p = Path(path)
        dn = display_name if display_name is not None else p.name
        return cls("upload", str(p), display_name=dn)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"JobSource(kind={self.kind!r}, ref={self.ref!r}, "
            f"display_name={self.display_name!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JobSource):
            return NotImplemented
        return (
            self.kind == other.kind
            and self.ref == other.ref
            and self.display_name == other.display_name
        )

    def __hash__(self) -> int:
        return hash((self.kind, self.ref, self.display_name))


# --- Public helpers ---------------------------------------------------------


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def audio_path_for(job_id: str) -> Path:
    """Return the on-disk path of a job's normalized source MP3."""
    return AUDIO_DIR / f"{job_id}.mp3"


def output_path_for(job_id: str) -> Path:
    """Return the on-disk path of a job's TTS narration MP3."""
    return OUTPUTS_DIR / f"{job_id}.mp3"


# --- Stage timing instrumentation (AC-10) ----------------------------------


@contextmanager
def _stage_timed(stage: str) -> Iterator[dict[str, float]]:
    """Time the wrapped block, stash the duration in ``out['seconds']``."""
    out: dict[str, float] = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield out
    finally:
        elapsed = time.perf_counter() - start
        out["seconds"] = round(elapsed, 3)
        logger.info("stage %s took %.3fs", stage, out["seconds"])


def _record_timing(
    stage: str,
    seconds: float,
    *,
    timings: dict[str, float],
) -> None:
    """Log a WARNING if ``seconds`` exceeds the per-stage soft budget."""
    timings[stage] = round(seconds, 3)
    budget = {
        "ingest": STAGE_BUDGET_INGEST_SECONDS,
        "transcribe": STAGE_BUDGET_TRANSCRIBE_SECONDS,
    }.get(stage, MAX_PIPELINE_SECONDS)
    if seconds > budget:
        logger.warning(
            "stage %s exceeded its soft budget: %.3fs > %.1fs (AC-10)",
            stage, seconds, budget,
        )


def _run_stage(
    stage: str,
    fn: Callable[[], "_T"],
    timings: dict[str, float],
) -> "_T":
    """Run a pipeline stage, time it, and record the result."""
    with _stage_timed(stage) as t:
        result = fn()
    _record_timing(stage, t["seconds"], timings=timings)
    return result


# --- Stage helpers ----------------------------------------------------------


def _stage_ingest(source: JobSource, job_id: str) -> Path:
    """Run the ingest stage (AC-2 or AC-3)."""
    if source.kind == "youtube":
        return ingest_mod.download_youtube(
            url=source.ref, job_id=job_id, output_dir=AUDIO_DIR
        )
    return ingest_mod.convert_upload_to_mp3(
        upload_path=source.ref, job_id=job_id, output_dir=AUDIO_DIR
    )


def _stage_transcribe(
    mp3_path: Path,
    transcribe_progress_callback: Callable[[int], None] | None = None,
) -> TranscriptionResult:
    """Run the transcribe stage (AC-4)."""
    return transcribe_mod.transcribe(
        mp3_path=mp3_path,
        transcribe_progress_callback=transcribe_progress_callback,
    )


# --- Public entry point -----------------------------------------------------


def run_job(
    job_id: str,
    source: JobSource,
    transcribe_progress_callback: Callable[[int], None] | None = None,
) -> TranscriptionResult:
    """Run the server-side pipeline and return a :class:`TranscriptionResult`.

    Two-stage process:
      1. Ingest — download YouTube audio or convert upload to MP3
      2. Transcribe — run Whisper for full transcript

    LLM extraction and TTS narration are handled on the frontend (BYOK).
    """
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("job_id must be a non-empty string.")
    if not isinstance(source, JobSource):
        raise ValueError(
            f"source must be a JobSource, got {type(source).__name__}"
        )

    logger.info("run_job start: job_id=%s source_kind=%s", job_id, source.kind)

    timings: dict[str, float] = {"ingest": 0.0, "transcribe": 0.0}

    # Stage 1: ingest
    mp3_path = _run_stage(
        "ingest", lambda: _stage_ingest(source, job_id), timings,
    )
    logger.info("run_job: ingest OK -> %s", mp3_path)

    # Stage 2: transcribe
    transcription = _run_stage(
        "transcribe",
        lambda: _stage_transcribe(mp3_path, transcribe_progress_callback),
        timings,
    )
    logger.info(
        "run_job: transcribe OK -> %d segments, %.1fs",
        len(transcription.segments), transcription.duration_seconds,
    )

    return transcription


# --- Async-friendly wrapper -------------------------------------------------


async def run_job_async(
    job_id: str,
    source: JobSource,
    transcribe_progress_callback: Callable[[int], None] | None = None,
) -> TranscriptionResult:
    """Async-friendly wrapper around :func:`run_job`."""
    import asyncio

    return await asyncio.to_thread(
        run_job, job_id, source, transcribe_progress_callback
    )


# --- Re-export --------------------------------------------------------------

__all__ = [
    "JobSource",
    "audio_path_for",
    "output_path_for",
    "run_job",
    "run_job_async",
    "utc_now_iso",
]
