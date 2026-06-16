"""Text-to-speech narration (AC-7).

Public entry point: :func:`narrate_to_mp3`.

Takes the AC-5 ``narrative`` string and synthesizes it to a playable
MP3 at ``data/outputs/<job_id>.mp3`` (AC-7). The default provider is
Edge TTS (``en-US-AriaNeural``) because it is free, fast, and
natural-sounding; OpenAI TTS is opt-in via ``TTS_PROVIDER=openai``.

Provider selection
------------------
* ``TTS_PROVIDER=edge`` (default) — uses the ``edge_tts`` package,
  which streams MP3 bytes from Microsoft's public Edge TTS endpoint
  and writes them to disk. No API key required.
* ``TTS_PROVIDER=openai`` — uses the OpenAI TTS API
  (``client.audio.speech.create``). Requires ``OPENAI_API_KEY``.
* ``TTS_PROVIDER=stub`` — emits a sine-wave MP3 via ``ffmpeg``. Used
  by tests that need a deterministic, network-free output; the
  actual file at ``data/outputs/<job_id>.mp3`` is still a real MP3
  any browser can play.

Duration validation
-------------------
AC-7 also requires "Narration duration within +/-15% of source video
duration." After synthesis we run ``ffprobe`` to measure the
narration's actual duration and compare it to
``source_duration_seconds`` (the transcription's ``duration_seconds``
— the same value the orchestrator has after the AC-4 stage). The
tolerance lives in :data:`app.config.NARRATION_DURATION_TOLERANCE`.

Failure modes
-------------
* The TTS provider raises (network down, auth failure, rate limit)
  -> :class:`TTSError` (HTTP 500, code ``TTS_FAILED``).
* The narration is empty -> :class:`ValueError`.
* The narration is outside the +/-15% window -> :class:`TTSError`
  with a clear message so the user can try a different voice /
  rate.
* ``ffprobe`` is missing or returns garbage -> :class:`TTSError`
  (defensive: AC-14 catches the missing-ffmpeg case at startup,
  but we re-check here in case the user uninstalled ffmpeg between
  boot and request).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Final

from ..config import NARRATION_DURATION_TOLERANCE

logger = logging.getLogger(__name__)


#: Default TTS voice. Mirrored in ``.env.example`` (TTS_VOICE).
DEFAULT_TTS_VOICE: Final[str] = "en-US-AriaNeural"

#: Default TTS provider. ``edge`` is the documented default.
DEFAULT_TTS_PROVIDER: Final[str] = "edge"

#: OpenAI TTS model and default voice when ``TTS_PROVIDER=openai``.
DEFAULT_OPENAI_TTS_MODEL: Final[str] = "tts-1"
DEFAULT_OPENAI_TTS_VOICE: Final[str] = "alloy"

#: Set of providers we know how to dispatch to. Anything else falls
#: back to ``edge`` with a logged warning.
_SUPPORTED_PROVIDERS: frozenset[str] = frozenset({"edge", "openai", "stub"})

#: Words-per-minute for the deterministic stub provider. Edge TTS
#: paces around 150-180 wpm; 150 is a good lower bound so tests can
#: assert on output duration.
STUB_WORDS_PER_MINUTE: Final[float] = 150.0

#: Floor / ceiling for the stub narration's duration.
STUB_MIN_SECONDS: Final[float] = 1.0
STUB_MAX_SECONDS: Final[float] = 90.0 * 60.0


class TTSError(Exception):
    """Raised when the TTS stage fails to produce a usable MP3.

    The API layer maps this to HTTP 500 with code ``TTS_FAILED``
    (see ``app/models.py``).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# --- Provider / voice resolution -------------------------------------------


def _select_provider(preferred: str | None = None) -> str:
    """Return the TTS provider to use, honoring arg > env > default."""
    requested = (
        preferred
        or os.environ.get("TTS_PROVIDER")
        or DEFAULT_TTS_PROVIDER
    ).lower()
    if requested not in _SUPPORTED_PROVIDERS:
        logger.warning(
            "Unknown TTS_PROVIDER=%r; falling back to %r",
            requested, DEFAULT_TTS_PROVIDER,
        )
        return DEFAULT_TTS_PROVIDER
    return requested


def _select_voice(voice: str | None = None, provider: str | None = None) -> str:
    """Return the TTS voice to use, honoring arg > env > default.

    The default differs by provider: Edge TTS uses
    ``en-US-AriaNeural``, OpenAI TTS uses ``alloy``.
    """
    if voice:
        return voice
    env_voice = os.environ.get("TTS_VOICE")
    if env_voice:
        return env_voice
    if (provider or _select_provider()) == "openai":
        return DEFAULT_OPENAI_TTS_VOICE
    return DEFAULT_TTS_VOICE


# --- Synthesis entry points -------------------------------------------------


async def _synthesize_edge_tts(
    text: str, voice: str, output_path: Path
) -> Path:
    """Synthesize ``text`` to MP3 using Microsoft Edge TTS.

    Wraps :class:`edge_tts.Communicate`. ``edge_tts`` is async, so the
    sync public entry point drives this via :func:`asyncio.run`.
    """
    try:
        import edge_tts
    except ImportError as exc:  # pragma: no cover
        raise TTSError(
            "edge_tts is not installed. Install it with "
            "`pip install edge-tts==6.1.18`."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        comm = edge_tts.Communicate(text, voice)
        await comm.save(str(output_path))
    except Exception as exc:
        raise TTSError(
            f"Edge TTS failed to synthesize {output_path.name!r}: {exc}"
        ) from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TTSError(
            f"Edge TTS reported success but produced no MP3 at "
            f"{output_path!s}."
        )
    return output_path


def _synthesize_openai_tts(
    text: str, voice: str, output_path: Path
) -> Path:
    """Synthesize ``text`` to MP3 using the OpenAI TTS API.

    ``OPENAI_API_KEY`` must be set in the environment.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise TTSError(
            "openai is not installed. Install it with "
            "`pip install openai==1.51.0`."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = OpenAI()
        model = os.environ.get("OPENAI_TTS_MODEL", DEFAULT_OPENAI_TTS_MODEL)
        response = client.audio.speech.create(
            model=model, voice=voice, input=text
        )
        # stream_to_file writes the binary audio content to disk.
        response.stream_to_file(str(output_path))
    except Exception as exc:
        raise TTSError(
            f"OpenAI TTS failed to synthesize {output_path.name!r}: {exc}"
        ) from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TTSError(
            f"OpenAI TTS reported success but produced no MP3 at "
            f"{output_path!s}."
        )
    return output_path


# --- Stub provider (deterministic, used by tests) --------------------------


def _stub_audio_duration_seconds(text: str) -> float:
    """Return the duration (s) the stub narration should be.

    Word-count / 150 wpm, clamped to the documented bounds.
    """
    word_count = max(1, len([w for w in text.split() if w.strip()]))
    raw = (word_count / STUB_WORDS_PER_MINUTE) * 60.0
    return max(STUB_MIN_SECONDS, min(STUB_MAX_SECONDS, raw))


def _ffmpeg_silence_cmd(duration_seconds: float, dst: Path) -> list[str]:
    """Build the ffmpeg argv for emitting a sine-wave MP3."""
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=44100:duration={duration_seconds}",
        "-ac", "1",
        "-b:a", "64k",
        "-f", "mp3",
        str(dst),
    ]


def _synthesize_stub_tts(text: str, output_path: Path) -> Path:
    """Emit a real MP3 at ``output_path`` whose length is text-driven.

    The output is a valid MP3 (44.1 kHz, mono, 64 kbps) that any
    browser can play. Tests can patch this function so CI never
    invokes the real TTS provider.
    """
    duration = _stub_audio_duration_seconds(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = _ffmpeg_silence_cmd(duration, output_path)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as exc:
        raise TTSError(
            "ffmpeg was not found on PATH. "
            "Install it with `apt-get install -y ffmpeg`."
        ) from exc
    except OSError as exc:  # pragma: no cover - defensive
        raise TTSError(
            f"Could not invoke ffmpeg to write {output_path}: {exc}"
        ) from exc

    if result.returncode != 0:
        stderr_tail = (result.stderr or "").strip().splitlines()[-3:]
        stderr_hint = ("\n".join(stderr_tail)).strip() or "<no stderr>"
        raise TTSError(
            f"ffmpeg exited with status {result.returncode} for TTS: "
            f"{stderr_hint}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TTSError(
            f"ffmpeg reported success but produced no MP3 at "
            f"{output_path!s}."
        )
    return output_path


# --- Dispatch + duration validation ---------------------------------------


def _dispatch_synthesis(
    provider: str, text: str, voice: str, output_path: Path
) -> Path:
    """Dispatch to the right provider's sync entry point.

    Edge TTS is async, so we drive it via :func:`asyncio.run`. The
    function is sync because the public entry point
    :func:`narrate_to_mp3` is sync (the orchestrator wraps the whole
    pipeline in ``asyncio.to_thread``).
    """
    if provider == "edge":
        return asyncio.run(
            _synthesize_edge_tts(text, voice, output_path)
        )
    if provider == "openai":
        return _synthesize_openai_tts(text, voice, output_path)
    if provider == "stub":
        return _synthesize_stub_tts(text, output_path)
    # _select_provider should never let an unknown value through.
    raise TTSError(f"Unknown TTS provider: {provider!r}")


# A regex that pulls the duration out of ffprobe's default JSON
# output. We use JSON because the alternative, ``-show_entries
# format=duration``, is fragile across ffmpeg versions.
_FFPROBE_DURATION_RE = re.compile(
    r'"duration"\s*:\s*"(?P<dur>[0-9]+(?:\.[0-9]+)?)"'
)


def _measure_mp3_duration_seconds(path: Path) -> float:
    """Return the duration of an MP3 file in seconds, via ffprobe.

    Raises:
        TTSError: ffprobe is missing, fails, or returns no duration.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as exc:
        raise TTSError(
            "ffprobe was not found on PATH. "
            "Install it with `apt-get install -y ffmpeg` (which "
            "ships ffprobe)."
        ) from exc

    if result.returncode != 0:
        raise TTSError(
            f"ffprobe exited with status {result.returncode} for "
            f"{path!s}: {(result.stderr or '').strip()}"
        )

    # Try JSON parse first; fall back to regex against the raw text.
    try:
        payload = json.loads(result.stdout or "{}")
        dur = float(payload.get("format", {}).get("duration", 0.0))
    except (ValueError, TypeError):
        m = _FFPROBE_DURATION_RE.search(result.stdout or "")
        if not m:
            raise TTSError(
                f"ffprobe did not report a duration for {path!s}."
            )
        dur = float(m.group("dur"))

    if dur <= 0.0:
        raise TTSError(
            f"ffprobe reported non-positive duration {dur}s for {path!s}."
        )
    return dur


def _check_duration_within_tolerance(
    narration_seconds: float, source_seconds: float
) -> None:
    """Raise :class:`TTSError` if the narration drifts > +/-15% of source.

    AC-7: "Narration duration within +/-15% of source video duration."
    """
    if source_seconds <= 0.0:
        # If we don't know the source duration, skip the check.
        return
    delta = abs(narration_seconds - source_seconds) / source_seconds
    if delta > NARRATION_DURATION_TOLERANCE:
        raise TTSError(
            f"Narration is {narration_seconds:.2f}s vs source "
            f"{source_seconds:.2f}s (drift {delta:.1%}); the plan "
            f"allows +/-{NARRATION_DURATION_TOLERANCE:.0%}. Try a "
            f"different voice or rate."
        )


# --- Public API -------------------------------------------------------------


def narrate_to_mp3(
    text: str,
    job_id: str,
    output_dir: Path | str,
    *,
    voice: str | None = None,
    provider: str | None = None,
    source_duration_seconds: float | None = None,
    check_duration: bool = True,
) -> Path:
    """Synthesize ``text`` to MP3 at ``output_dir/<job_id>.mp3`` (AC-7).

    Args:
        text: The narrative text (from AC-5 ``KnowledgeExtract.narrative``).
        job_id: Unique identifier for this job; used as the filename.
        output_dir: Directory the MP3 will be written to. Created if it
            does not exist.
        voice: Optional TTS voice override. Defaults to the value of
            the ``TTS_VOICE`` env var, then a per-provider default
            (``en-US-AriaNeural`` for Edge, ``alloy`` for OpenAI).
        provider: Optional TTS provider override (``edge``, ``openai``,
            or ``stub``). Defaults to the ``TTS_PROVIDER`` env var, then
            :data:`DEFAULT_TTS_PROVIDER` (``"edge"``).
        source_duration_seconds: If provided (and ``check_duration`` is
            True), the function will fail if the produced narration is
            outside +/-15% of this value. Pass the transcription's
            ``duration_seconds`` (AC-4) to validate against the source
            video length.
        check_duration: If False, skip the +/-15% check. Useful for tests
            that don't care about pacing.

    Returns:
        The :class:`pathlib.Path` of the resulting MP3 file.

    Raises:
        ValueError: ``text`` is empty.
        TTSError: Synthesis failed, the file is invalid, or the
            narration drift exceeds :data:`app.config.NARRATION_DURATION_TOLERANCE`.
    """
    if not text or not text.strip():
        raise ValueError("Cannot narrate empty text.")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("job_id must be a non-empty string.")

    chosen_provider = _select_provider(provider)
    chosen_voice = _select_voice(voice, chosen_provider)
    logger.debug(
        "narrate_to_mp3: job_id=%s provider=%s voice=%s",
        job_id, chosen_provider, chosen_voice,
    )

    out = Path(output_dir)
    final_path = out / f"{job_id}.mp3"
    _dispatch_synthesis(chosen_provider, text, chosen_voice, final_path)

    if check_duration and source_duration_seconds is not None:
        actual = _measure_mp3_duration_seconds(final_path)
        _check_duration_within_tolerance(actual, source_duration_seconds)

    return final_path


def public_audio_url(job_id: str) -> str:
    """Return the public URL of a job's narration MP3 (AC-7).

    Kept as a tiny helper so the orchestrator and the API route share
    one definition of the URL shape (``/audio/<job_id>.mp3``).
    """
    return f"/audio/{job_id}.mp3"
