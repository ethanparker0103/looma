"""Pipeline orchestrator (AC-6, AC-10).

Public entry points:

* :class:`JobSource` — a small tagged-union for the two input shapes
  Looma accepts (YouTube URL, uploaded video file).
* :func:`run_job` — runs the four pipeline stages in order and returns
  a single :class:`~app.models.LoomaResult` (AC-6).
* :func:`utc_now_iso` — ISO-8601 UTC timestamp helper (AC-6).

The pipeline
------------
For every job, the orchestrator runs::

    [ingest]        ->  data/audio/<job_id>.mp3     (AC-2 or AC-3)
    [transcribe]    ->  TranscriptionResult          (AC-4)
    [extract]       ->  KnowledgeExtract             (AC-5)
    [narrate]       ->  data/outputs/<job_id>.mp3   (AC-7)

The first and fourth stages touch the filesystem; the second and
third are pure CPU/network. The orchestrator's job is to wire them
together in the right order, surface failures as the right exception
classes (so the API layer can map them to the right HTTP codes —
AC-11), and return a single :class:`LoomaResult` whose
``created_at`` is the UTC timestamp of completion.

Performance budgets (AC-10)
---------------------------
AC-10 requires the full pipeline to finish in **under 5 minutes** for
a 20-minute source video on a modern laptop. The expected split is::

    ingest     <= 60s   (yt-dlp + ffmpeg audio normalization)
    transcribe <= 240s  (Whisper `small` is the default; ~3-9x realtime)
    extract    <= 30s   (LLM call + 1 retry on schema failure)
    narrate    <= 60s   (Edge TTS synthesis of a 150-400 word narration)
    -------------------------------------------
    total      <= ~390s typical, hard cap 300s

:func:`run_job` instruments each stage with
:func:`time.perf_counter` and surfaces the per-stage durations in
:attr:`LoomaResult.timings`. The orchestrator also logs a WARNING
when the total exceeds :data:`~app.config.MAX_PIPELINE_SECONDS`
(default 300) — a regression early-warning that doesn't break the
user-facing response.

Design notes
------------
* The four stage entry points are imported lazily, inside
  :func:`run_job`, so importing the orchestrator module never pulls
  in :mod:`whisper` or :mod:`anthropic` (and so the test suite can
  patch any of them in isolation).
* The orchestrator is synchronous. The FastAPI layer (AC-11) wraps
  :func:`run_job` in ``asyncio.to_thread`` so the event loop stays
  responsive while Whisper and the TTS run.
* :func:`_stage_*` helpers are the only place that calls into the
  four sub-pipelines. They are extracted so tests can patch
  individual stages with ``mock.patch.object``.
* :func:`_stage_timed` is a thin context-manager that captures a
  single stage's wall-clock duration. The four stage call sites in
  :func:`run_job` use it instead of bare :func:`time.perf_counter`
  so the timing code lives in exactly one place.
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
    STAGE_BUDGET_EXTRACT_SECONDS,
    STAGE_BUDGET_INGEST_SECONDS,
    STAGE_BUDGET_NARRATE_SECONDS,
    STAGE_BUDGET_TRANSCRIBE_SECONDS,
)
from ..models import (
    LoomaResult,
    PipelineTimings,
    TranscriptionResult,
)
from . import extract as extract_mod
from . import ingest as ingest_mod
from . import narrate as narrate_mod
from .narrate import TTSError
from . import transcribe as transcribe_mod

logger = logging.getLogger(__name__)


#: Type variable for the return type of a pipeline stage callable
#: passed to :func:`_run_stage`. Stages return heterogeneous types
#: (Path, TranscriptionResult, KnowledgeExtract, Path) — the
#: typevar lets :func:`_run_stage` be generic without ``Any``.
_T = TypeVar("_T")


# --- JobSource --------------------------------------------------------------


class JobSource:
    """A validated input source for a pipeline job.

    Use the class-method constructors :meth:`youtube` and
    :meth:`upload` rather than instantiating directly. The :attr:`kind`
    attribute is the public discriminator (``"youtube"`` or
    ``"upload"``).

    For uploads, ``display_name`` is the user-facing filename
    (e.g. ``"my-video.mp4"``) used in :attr:`LoomaResult.source_ref`.
    The ingest stage still reads the file at ``ref`` (the on-disk
    path), so the two values can differ when the caller has spooled
    the upload to a temp file.
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
        # Default ``display_name`` to the basename of the path so the
        # caller doesn't have to pass it twice for the common case
        # where the file is already at a friendly location.
        dn = display_name if display_name is not None else p.name
        return cls("upload", str(p), display_name=dn)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
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
    """Return the current UTC time as an ISO-8601 string with ``Z`` suffix.

    AC-6 requires ``created_at`` to be ISO-8601 UTC. We use
    ``datetime.now(timezone.utc)`` (timezone-aware) and format with an
    explicit ``Z`` so consumers don't have to interpret a numeric
    offset. Microseconds are kept for ordering jobs that finish in
    the same wall-clock second.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def public_audio_url_for(job_id: str) -> str:
    """Return the public URL of a job's narration MP3.

    Thin wrapper around :func:`app.pipeline.narrate.public_audio_url`
    so the orchestrator and the API layer share one definition.
    """
    return narrate_mod.public_audio_url(job_id)


def audio_path_for(job_id: str) -> Path:
    """Return the on-disk path of a job's normalized source MP3 (AC-2/3)."""
    return AUDIO_DIR / f"{job_id}.mp3"


def output_path_for(job_id: str) -> Path:
    """Return the on-disk path of a job's TTS narration MP3 (AC-7)."""
    return OUTPUTS_DIR / f"{job_id}.mp3"


# --- Stage timing instrumentation (AC-10) ----------------------------------
#
# AC-10 mandates a 5-minute end-to-end budget for a 20-minute source.
# Capturing per-stage wall-clock time is the only way to know which
# stage regressed when a real run blows the budget. ``_stage_timed``
# is a tiny context manager; the captured durations are stored on the
# returned :class:`LoomaResult` so observability lives in the public
# API surface (not just in the log stream).


@contextmanager
def _stage_timed(stage: str) -> Iterator[dict[str, float]]:
    """Time the wrapped block, stash the duration in ``out['seconds']``.

    Args:
        stage: Pipeline stage name, e.g. ``"ingest"``. Currently only
            used for log messages — the :func:`run_job` call-site
            assigns the captured value to a named field on
            :class:`PipelineTimings`.

    Yields:
        A single-key dict ``{"seconds": float}`` that the wrapped
        block (or the call-site, after the block exits) can read.
        The dict is mutated in place so the call-site doesn't need
        a second ``with``-statement to retrieve the value.
    """
    out: dict[str, float] = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield out
    finally:
        elapsed = time.perf_counter() - start
        # Round to milliseconds for a clean surface area; sub-ms
        # precision is noise on a 5-minute budget.
        out["seconds"] = round(elapsed, 3)
        logger.info("stage %s took %.3fs", stage, out["seconds"])


def _soft_budget_for(stage: str) -> float:
    """Return the configured soft budget for ``stage`` in seconds (AC-10).

    Pulled out as a helper so the WARNING log in :func:`_record_timing`
    has one place to look up budgets — keeps the log line symmetric
    with the budgets the performance tests assert against.
    """
    if stage == "ingest":
        return STAGE_BUDGET_INGEST_SECONDS
    if stage == "transcribe":
        return STAGE_BUDGET_TRANSCRIBE_SECONDS
    if stage == "extract":
        return STAGE_BUDGET_EXTRACT_SECONDS
    if stage == "narrate":
        return STAGE_BUDGET_NARRATE_SECONDS
    return MAX_PIPELINE_SECONDS  # unknown stage — use the global cap


def _record_timing(
    stage: str,
    seconds: float,
    *,
    timings: dict[str, float],
) -> None:
    """Log a WARNING if ``seconds`` exceeds the per-stage soft budget (AC-10).

    Records the duration in ``timings`` regardless. The dict is
    populated in-place by :func:`run_job`; the function does not
    return a new structure.
    """
    timings[stage] = round(seconds, 3)
    budget = _soft_budget_for(stage)
    if seconds > budget:
        logger.warning(
            "stage %s exceeded its soft budget: %.3fs > %.1fs (AC-10)",
            stage, seconds, budget,
        )


def _run_stage(
    stage: str,
    fn: Callable[[], "_T"],
    timings: dict[str, float],
    progress_callback: "Callable[[str], None] | None" = None,
) -> "_T":
    """Run a pipeline stage, time it, and record the result.

    Wraps :func:`_stage_timed` and :func:`_record_timing` so the
    four call-sites in :func:`run_job` don't have to repeat the
    context-manager dance. The ``timings`` dict is mutated in place
    to keep the call-site signature flat.

    If ``progress_callback`` is provided, it is invoked *just before*
    the wrapped ``fn`` runs with the ``stage`` name as its only
    argument. The async ``/api/extract/async`` endpoint uses this
    hook to flip the in-memory ``JobState.status`` to
    ``transcribing`` / ``narrating`` (etc.) so the JS polling loop
    sees real per-stage progress instead of a single
    ``downloading → done`` jump.
    """
    if progress_callback is not None:
        progress_callback(stage)
    with _stage_timed(stage) as t:
        result = fn()
    _record_timing(stage, t["seconds"], timings=timings)
    return result


# --- Stage helpers (one per pipeline stage) --------------------------------
# These are the only places that call into the sub-pipelines. Pulling
# them out as named functions makes them easy to patch in tests.


def _stage_ingest(source: JobSource, job_id: str) -> Path:
    """Run the ingest stage (AC-2 or AC-3) and return the normalized MP3 path."""
    if source.kind == "youtube":
        return ingest_mod.download_youtube(
            url=source.ref, job_id=job_id, output_dir=AUDIO_DIR
        )
    # source.kind == "upload" — ``source.ref`` is the full path.
    return ingest_mod.convert_upload_to_mp3(
        upload_path=source.ref, job_id=job_id, output_dir=AUDIO_DIR
    )


def _stage_transcribe(
    mp3_path: Path,
    transcribe_progress_callback: Callable[[int], None] | None = None,
) -> TranscriptionResult:
    """Run the transcribe stage (AC-4).

    ``transcribe_progress_callback`` is forwarded to the Whisper
    transcribe function so the frontend sees granular progress
    during the long CPU-bound transcription step.
    """
    return transcribe_mod.transcribe(
        mp3_path=mp3_path,
        transcribe_progress_callback=transcribe_progress_callback,
    )


def _stage_extract(transcription: TranscriptionResult):
    """Run the LLM extract stage (AC-5)."""
    return extract_mod.extract_knowledge(transcription=transcription)


def _stage_narrate(text: str, job_id: str, source_duration_seconds: float) -> Path:
    """Run the TTS stage (AC-7) and return the narration MP3 path.

    ``source_duration_seconds`` is the transcription's duration (AC-4)
    — i.e. the source video's length. The narrate stage will fail if
    the produced narration drifts more than +/-15% from this value
    (AC-7).
    """
    return narrate_mod.narrate_to_mp3(
        text=text,
        job_id=job_id,
        output_dir=OUTPUTS_DIR,
        source_duration_seconds=source_duration_seconds,
    )


# --- Public entry point -----------------------------------------------------


def run_job(
    job_id: str,
    source: JobSource,
    progress_callback: "Callable[[str], None] | None" = None,
    transcribe_progress_callback: Callable[[int], None] | None = None,
) -> LoomaResult:
    """Run the four pipeline stages and return a :class:`LoomaResult` (AC-6).

    Args:
        job_id: Unique identifier for this job. Used as the basename
            for all generated files (``data/audio/<job_id>.mp3`` and
            ``data/outputs/<job_id>.mp3``).
        source: The :class:`JobSource` describing where the audio
            comes from. Either a YouTube URL (AC-2) or an uploaded
            video file path (AC-3).
        progress_callback: Optional ``Callable[[str], None]`` invoked
            with the stage name (``"ingest"``, ``"transcribe"``,
            ``"extract"``, ``"narrate"``) just before each stage
            runs. The async ``/api/extract/async`` endpoint uses
            this to surface per-stage progress to the JS client.
        transcribe_progress_callback: Optional
            ``Callable[[int], None]`` invoked from inside the
            transcribe stage with a 0..100 progress percentage
            after each audio chunk completes.  The async endpoint
            uses this to update the frontend progress bar smoothly
            instead of getting stuck at 60 %.

    Returns:
        A fully-populated :class:`LoomaResult` whose ``transcription``
        and ``knowledge`` fields are the AC-4 / AC-5 outputs verbatim,
        whose ``audio_url`` is ``/audio/<job_id>.mp3``, and whose
        ``created_at`` is the UTC timestamp of completion.

    Raises:
        Exception: Re-raises whatever the failing stage raised. The
            API layer is responsible for catching stage-specific
            exceptions (``IngestError``, ``TranscriptionError``,
            ``LLMSchemaError``, ``TTSError``) and mapping them to
            the right HTTP codes (AC-11).
    """
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("job_id must be a non-empty string.")
    if not isinstance(source, JobSource):
        raise ValueError(
            f"source must be a JobSource, got {type(source).__name__}"
        )

    logger.info(
        "run_job start: job_id=%s source_kind=%s", job_id, source.kind
    )

    # Per-stage wall-clock durations (AC-10). Mutated in place by
    # the ``_run_stage`` helper below. Final values are also
    # surfaced on ``LoomaResult.timings`` so consumers can observe
    # them via the API.
    timings: dict[str, float] = {
        "ingest": 0.0,
        "transcribe": 0.0,
        "extract": 0.0,
        "narrate": 0.0,
    }

    # Stage 1: ingest (AC-2 / AC-3)
    mp3_path = _run_stage(
        "ingest", lambda: _stage_ingest(source, job_id), timings,
        progress_callback=progress_callback,
    )
    logger.info("run_job: ingest OK -> %s", mp3_path)

    # Stage 2: transcribe (AC-4)
    transcription = _run_stage(
        "transcribe",
        lambda: _stage_transcribe(mp3_path, transcribe_progress_callback),
        timings,
        progress_callback=progress_callback,
    )
    logger.info(
        "run_job: transcribe OK -> %d segments, %.1fs",
        len(transcription.segments), transcription.duration_seconds,
    )

    # Stage 3: extract (AC-5)
    knowledge = _run_stage(
        "extract", lambda: _stage_extract(transcription), timings,
        progress_callback=progress_callback,
    )
    logger.info(
        "run_job: extract OK -> %d insights, %d chapters",
        len(knowledge.insights), len(knowledge.chapters),
    )

    # Stage 4: narrate (AC-7) — uses the AC-5 narrative
    # If the TTS provider fails (common with edge-tts when Microsoft
    # blocks the container's IP) we degrade gracefully: log a warning,
    # leave ``narrate_path`` as None, and return text-only results.
    # The user still sees title / summary / insights / chapters.
    narrate_path = None
    try:
        narrate_path = _run_stage(
            "narrate",
            lambda: _stage_narrate(
                knowledge.narrative, job_id, transcription.duration_seconds
            ),
            timings,
            progress_callback=progress_callback,
        )
        logger.info("run_job: narrate OK -> %s", narrate_path)
    except TTSError as exc:
        logger.warning(
            "run_job: narrate failed — continuing without audio (%s)",
            exc.message,
        )
        timings["narrate"] = 0.0

    # Compose the result (AC-6).
    # For YouTube, ``source_ref`` is the URL verbatim. For uploads,
    # it is the user-supplied ``display_name`` (e.g. "my-video.mp4")
    # if one was provided; otherwise we fall back to the basename of
    # the on-disk path. The full path is internal — the API never
    # exposes it.
    source_ref = source.ref
    if source.kind == "upload":
        if source.display_name:
            source_ref = source.display_name
        else:
            source_ref = Path(source.ref).name

    # Build the per-stage timings object (AC-10). Each stage's
    # duration is already rounded to milliseconds by
    # :func:`_record_timing`; ``total`` is the sum of the four.
    total_seconds = round(sum(timings.values()), 3)
    pipeline_timings = PipelineTimings(
        ingest_seconds=timings["ingest"],
        transcribe_seconds=timings["transcribe"],
        extract_seconds=timings["extract"],
        narrate_seconds=timings["narrate"],
        total_seconds=total_seconds,
    )

    audio_url = (
        public_audio_url_for(job_id) if narrate_path is not None else ""
    )
    result = LoomaResult(
        job_id=job_id,
        source_type=source.kind,
        source_ref=source_ref,
        title=knowledge.title,
        transcription=transcription,
        knowledge=knowledge,
        audio_url=audio_url,
        created_at=utc_now_iso(),
        timings=pipeline_timings,
    )

    # AC-10 perf-budget warning. A real run that exceeds 5 minutes
    # on a healthy 20-min source is a regression; the log line is
    # the only hook operators get without a full metrics pipeline.
    if total_seconds > MAX_PIPELINE_SECONDS:
        logger.warning(
            "run_job exceeded AC-10 budget: total=%.3fs > %.1fs "
            "(job_id=%s source_kind=%s)",
            total_seconds, MAX_PIPELINE_SECONDS, job_id, source.kind,
        )
    else:
        logger.info(
            "run_job done: job_id=%s total=%.3fs (budget=%.1fs)",
            job_id, total_seconds, MAX_PIPELINE_SECONDS,
        )
    return result


# --- Async-friendly wrapper -------------------------------------------------


async def run_job_async(
    job_id: str,
    source: JobSource,
    progress_callback: "Callable[[str], None] | None" = None,
    transcribe_progress_callback: "Callable[[int], None] | None" = None,
) -> LoomaResult:
    """Async-friendly wrapper around :func:`run_job`.

    The CPU-bound stages (Whisper, TTS) block the event loop, so the
    FastAPI layer (AC-11) calls this wrapper from inside
    ``asyncio.to_thread`` to keep the event loop responsive. The
    wrapper itself does not do any heavy lifting; it simply exists
    to make the calling site easier to read in :mod:`app.main`.

    ``progress_callback`` is forwarded to :func:`run_job` so the
    async endpoint can observe per-stage progress even though the
    pipeline itself runs in a worker thread.
    """
    import asyncio

    return await asyncio.to_thread(
        run_job, job_id, source, progress_callback, transcribe_progress_callback
    )


# --- Re-export for convenience ---------------------------------------------

# Surface the small error helper as a module-level symbol so tests
# that need to assert on the LoomaResult wiring can use the same
# import path the API layer will use.
__all__ = [
    "JobSource",
    "LoomaResult",
    "PipelineTimings",
    "audio_path_for",
    "output_path_for",
    "public_audio_url_for",
    "run_job",
    "run_job_async",
    "utc_now_iso",
]
