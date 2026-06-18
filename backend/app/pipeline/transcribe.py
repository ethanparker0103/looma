"""Whisper transcription (AC-4).

Public entry point: :func:`transcribe`.

Loads ``openai-whisper`` (configurable model size, default ``small``)
and runs it on the normalized MP3, returning a strict
:class:`~app.models.TranscriptionResult` with the full transcript,
time-stamped segments, detected language, and total duration.

For long audio files (> 60 s) the input is split into chunks and
transcribed piece by piece, with a ``transcribe_progress_callback``
fired after each chunk so the frontend sees granular progress instead
of getting stuck at 60 % for minutes.

Design notes
------------
* The Whisper model is heavy (5-30 s cold-load for ``small``) so
  :func:`_load_model` is ``functools.lru_cache``d. The first call
  pays the penalty; subsequent calls reuse the same model object.
* :func:`transcribe` is synchronous because ``whisper.transcribe`` is
  CPU-bound; the orchestrator wraps it in ``asyncio.to_thread`` so
  the FastAPI event loop stays responsive.
* The conversion from Whisper's dict-shaped result to our Pydantic
  schema is the only place the rest of the codebase touches Whisper
  internals - swapping the ASR backend later means rewriting this
  one function.
* Chunked transcription uses ffprobe to get audio duration and ffmpeg
  to extract segments. Both tools are guaranteed to be on PATH by the
  AC-14 startup guard in ``app.main``.
"""

from __future__ import annotations

import json
import os
import subprocess as _subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import whisper

from ..models import TranscriptSegment, TranscriptionResult


#: Default Whisper model size if ``WHISPER_MODEL`` is unset. ``small``
#: is a good accuracy/speed trade-off; override via env to use
#: ``tiny|base|small|medium|large``.
DEFAULT_WHISPER_MODEL: str = "small"

#: Whisper's ``transcribe`` returns the language as a code like ``"en"``
#: or ``"fr"". If it somehow comes back empty we fall back to this.
DEFAULT_LANGUAGE: str = "en"

#: Max audio duration (seconds) for a single Whisper transcribe call.
#: Audio files shorter than this threshold are transcribed in one
#: shot (no chunking).  Files longer than this are split into
#: chunks of approximately this size so the progress callback fires
#: more frequently and the frontend bar moves smoothly.
_CHUNK_MAX_SECONDS = 20

#: Maximum number of chunks to split into.  20 chunks x 20 s/chunk
#: covers ~400 s of audio (6+ minutes), which at ~3-9x realtime on
#: CPU means each chunk takes about 1-3 minutes of wall-clock.
_MAX_CHUNKS = 20


class TranscriptionError(Exception):
    """Raised when Whisper fails to produce a usable result.

    The API layer maps this to HTTP 500 with code
    ``TRANSCRIPTION_FAILED`` (see ``app/models.py``).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> Any:
    """Load the Whisper model, caching by name.

    Caching is keyed on ``model_name`` so tests that exercise a
    different model size don't collide with the production cache. The
    cache is bounded at 4 entries -- enough for the common sizes
    (``tiny|base|small|medium|large``) without holding multiple
    gigabytes of weights in memory indefinitely.
    """
    return whisper.load_model(model_name)


def _resolve_model_name(model_name: str | None) -> str:
    """Return the model name to use, honoring the explicit arg then env."""
    if model_name:
        return model_name
    return os.environ.get("WHISPER_MODEL", DEFAULT_WHISPER_MODEL)


def _duration_from_segments(segments: list[TranscriptSegment]) -> float:
    """Return the audio duration in seconds, derived from the last segment.

    Whisper guarantees ``segments`` is sorted by ``start`` ascending, so
    the end of the last segment is the duration. If there are no
    segments (silent / empty audio) we report ``0.0``.
    """
    if not segments:
        return 0.0
    return float(segments[-1].end)


def _get_audio_duration_ffprobe(path: Path) -> float:
    """Return the audio duration in seconds using ``ffprobe``.

    This is more reliable than stat() on variable-bitrate MP3s.  The
    ``ffprobe`` binary is guaranteed on PATH by the AC-14 startup guard.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(path),
    ]
    try:
        out = _subprocess.check_output(cmd, text=True, stderr=_subprocess.DEVNULL)
        info = json.loads(out)
        return float(info.get("format", {}).get("duration", 0.0))
    except Exception:  # noqa: BLE001 - best-effort fallback
        return 0.0


def _extract_audio_segment(
    src: Path, start: float, end: float, tmpdir: Path
) -> Path:
    """Extract ``[start, end)`` from ``src`` to a temp WAV file.

    The segment is written as 16-bit mono WAV at 16 kHz so Whisper
    can read it directly without re-decoding the whole file.
    """
    seg = tmpdir / f"seg_{start:.0f}_{end:.0f}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "quiet",
        "-i", str(src),
        "-ss", str(start),
        "-to", str(end),
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        str(seg),
    ]
    _subprocess.check_call(cmd, stderr=_subprocess.DEVNULL)
    return seg


def _merge_segments(
    results: list[dict],
) -> tuple[list[dict], str]:
    """Merge chunk transcription results into a single transcript dict.

    Returns ``(segments, text)`` where ``segments`` is a list of
    Whisper-style segment dicts (``start``, ``end``, ``text``) with
    monotonically increasing timestamps and ``text`` is the joined
    plain-text transcript.
    """
    all_segments: list[dict] = []
    text_parts: list[str] = []
    offset = 0.0
    for chunk in results:
        for seg in (chunk.get("segments") or []):
            seg["start"] = float(seg.get("start", 0.0)) + offset
            seg["end"] = float(seg.get("end", 0.0)) + offset
            all_segments.append(seg)
            text_parts.append(str(seg.get("text", "")).strip())
        # Accumulate offset: the next chunk's timestamps are relative
        # to the start of that chunk, so we shift by the chunk duration.
        offset = all_segments[-1]["end"] if all_segments else offset
    return all_segments, " ".join(text_parts)


def transcribe(
    mp3_path: Path | str,
    model_name: str | None = None,
    *,
    transcribe_progress_callback: Callable[[int], None] | None = None,
) -> TranscriptionResult:
    """Transcribe a normalized MP3 to a :class:`TranscriptionResult`.

    Args:
        mp3_path: Path to the normalized MP3 (the output of
            :func:`app.pipeline.ingest.download_youtube` or
            :func:`app.pipeline.ingest.convert_upload_to_mp3`).
        model_name: Optional override for the Whisper model size. If
            ``None``, the ``WHISPER_MODEL`` env var is consulted; if
            that is also unset, :data:`DEFAULT_WHISPER_MODEL` is used.
        transcribe_progress_callback: Optional callback fired after
            each audio chunk is transcribed.  The argument is the
            overall progress percentage (``0`` .. ``100``) of the
            transcribe stage.  The caller (``app.main``) uses this
            to produce granular progress updates for the frontend
            progress bar.

    Returns:
        A :class:`TranscriptionResult` containing the full transcript,
        ordered time-stamped segments, the detected language code, and
        the total audio duration in seconds.

    Raises:
        FileNotFoundError: ``mp3_path`` does not exist on disk.
        TranscriptionError: Whisper raised an exception while
            transcribing the file. The original exception is chained.
    """
    src = Path(mp3_path)
    if not src.exists():
        raise FileNotFoundError(f"MP3 not found: {src!s}")

    name = _resolve_model_name(model_name)
    model = _load_model(name)

    total_duration = _get_audio_duration_ffprobe(src)

    # For short files (< 60 s) or when we can't get duration, just
    # transcribe in one shot.  The frontend will see a single jump to
    # 60 / 85 % which is acceptable for quick jobs.
    if total_duration <= _CHUNK_MAX_SECONDS or total_duration <= 0:
        return _transcribe_short(src, model, name)

    # Chunked transcription for long files.
    return _transcribe_chunked(
        src, model, name, total_duration,
        transcribe_progress_callback=transcribe_progress_callback,
    )


def _transcribe_short(
    src: Path, model: Any, name: str,
) -> TranscriptionResult:
    """Transcribe a short audio file in a single shot."""
    try:
        raw: dict[str, Any] = model.transcribe(str(src), verbose=False)
    except Exception as exc:
        raise TranscriptionError(
            f"Whisper failed to transcribe {src.name!r} "
            f"with model {name!r}: {exc}"
        ) from exc

    segments = _whisper_raw_to_segments(raw)
    language = str(raw.get("language") or DEFAULT_LANGUAGE)
    transcript = str(raw.get("text") or "").strip()
    return TranscriptionResult(
        transcript=transcript,
        segments=segments,
        language=language,
        duration_seconds=_duration_from_segments(segments),
    )


def _transcribe_chunked(
    src: Path, model: Any, name: str,
    total_duration: float,
    *,
    transcribe_progress_callback: Callable[[int], None] | None = None,
) -> TranscriptionResult:
    """Divide the audio into chunks and transcribe each independently."""
    chunk_count = min(_MAX_CHUNKS, max(2, int(total_duration / _CHUNK_MAX_SECONDS)))
    chunk_duration = total_duration / chunk_count
    chunk_results: list[dict] = []
    detected_language: str = DEFAULT_LANGUAGE

    with tempfile.TemporaryDirectory(prefix="whisper-chunks-") as tmp:
        tmpdir = Path(tmp)
        for i in range(chunk_count):
            start = i * chunk_duration
            end = (i + 1) * chunk_duration if i < chunk_count - 1 else total_duration

            seg_path = _extract_audio_segment(src, start, end, tmpdir)
            try:
                chunk_raw = model.transcribe(str(seg_path), verbose=False)
            except Exception as exc:
                raise TranscriptionError(
                    f"Whisper failed on chunk {i + 1}/{chunk_count} "
                    f"of {src.name!r} with model {name!r}: {exc}"
                ) from exc

            chunk_results.append(chunk_raw)
            if i == 0:
                detected_language = str(
                    chunk_raw.get("language") or DEFAULT_LANGUAGE
                )

            if transcribe_progress_callback is not None:
                pct = int((i + 1) / chunk_count * 100)
                transcribe_progress_callback(pct)

    merged_segments, merged_text = _merge_segments(chunk_results)
    merged_segments_pydantic = [
        TranscriptSegment(
            start=float(s.get("start", 0.0)),
            end=float(s.get("end", 0.0)),
            text=str(s.get("text", "")).strip(),
        )
        for s in merged_segments
    ]
    merged_segments_pydantic.sort(key=lambda s: (s.start, s.end))

    return TranscriptionResult(
        transcript=merged_text,
        segments=merged_segments_pydantic,
        language=detected_language,
        duration_seconds=_duration_from_segments(merged_segments_pydantic),
    )


def _whisper_raw_to_segments(raw: dict[str, Any]) -> list[TranscriptSegment]:
    """Convert Whisper's raw output dict to a list of TranscriptSegment.

    Shared by the single-shot and chunked code paths.
    """
    raw_segments = raw.get("segments") or []
    segments = [
        TranscriptSegment(
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
            text=str(seg.get("text", "")).strip(),
        )
        for seg in raw_segments
    ]
    segments.sort(key=lambda s: (s.start, s.end))
    return segments
