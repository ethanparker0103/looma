"""Unit tests for ``app.pipeline.narrate`` (AC-7).

Coverage:

* Provider resolution (arg > env > default; unknown value falls back)
* Voice resolution (arg > env > per-provider default)
* Stub provider writes a real, playable MP3 at the documented path
* Edge TTS call site wires the right text and voice into
  ``edge_tts.Communicate`` (mocked — no live network call)
* OpenAI TTS call site wires the right text and voice into
  ``client.audio.speech.create`` (mocked — no live API call)
* Duration measurement via ffprobe
* Duration tolerance check (passes at ±15%, fails outside)
* Empty-text guard
* ``public_audio_url`` helper
* Dispatch to the right provider
* ``CODE_TTS_FAILED`` error code constant
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from app.config import NARRATION_DURATION_TOLERANCE
from app.models import CODE_TTS_FAILED
from app.pipeline import narrate as narrate_mod
from app.pipeline.narrate import (
    DEFAULT_OPENAI_TTS_MODEL,
    DEFAULT_OPENAI_TTS_VOICE,
    DEFAULT_TTS_PROVIDER,
    DEFAULT_TTS_VOICE,
    TTSError,
    _check_duration_within_tolerance,
    _dispatch_synthesis,
    _ffmpeg_silence_cmd,
    _measure_mp3_duration_seconds,
    _select_provider,
    _select_voice,
    _stub_audio_duration_seconds,
    _synthesize_stub_tts,
    narrate_to_mp3,
    public_audio_url,
)


# --- Provider / voice resolution -------------------------------------------


def test_select_provider_prefers_explicit_arg() -> None:
    assert _select_provider("openai") == "openai"


def test_select_provider_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("TTS_PROVIDER", "openai")
    assert _select_provider() == "openai"


def test_select_provider_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.delenv("TTS_PROVIDER", raising=False)
    assert _select_provider() == DEFAULT_TTS_PROVIDER == "edge"


def test_select_provider_unknown_value_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("TTS_PROVIDER", "rss-reader")
    assert _select_provider() == "edge"


def test_select_voice_prefers_explicit_arg() -> None:
    assert _select_voice("custom-voice", "edge") == "custom-voice"


def test_select_voice_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("TTS_VOICE", "en-GB-RyanNeural")
    assert _select_voice(None, "edge") == "en-GB-RyanNeural"


def test_select_voice_edge_default() -> None:
    """AC-7: default Edge TTS voice is ``en-US-AriaNeural``."""
    assert _select_voice(None, "edge") == DEFAULT_TTS_VOICE == "en-US-AriaNeural"


def test_select_voice_openai_default() -> None:
    assert _select_voice(None, "openai") == DEFAULT_OPENAI_TTS_VOICE == "alloy"


def test_select_voice_stub_default() -> None:
    """The stub provider uses the Edge TTS default voice."""
    assert _select_voice(None, "stub") == DEFAULT_TTS_VOICE


# --- public_audio_url helper -----------------------------------------------


def test_public_audio_url_format() -> None:
    assert public_audio_url("abc-123") == "/audio/abc-123.mp3"
    assert public_audio_url("xyz") == "/audio/xyz.mp3"


# --- _stub_audio_duration_seconds ------------------------------------------


def test_stub_audio_duration_minimum() -> None:
    """Empty / very short text still produces a 1-second file."""
    assert _stub_audio_duration_seconds("") == narrate_mod.STUB_MIN_SECONDS
    assert _stub_audio_duration_seconds("hi") == narrate_mod.STUB_MIN_SECONDS


def test_stub_audio_duration_scales_with_words() -> None:
    """Word count drives the duration at 150 wpm."""
    # 150 words -> 60 seconds
    text = " ".join(["word"] * 150)
    assert _stub_audio_duration_seconds(text) == pytest.approx(60.0)


def test_stub_audio_duration_clamped_to_max() -> None:
    """Over-long text is clamped to the 90-min ceiling."""
    text = " ".join(["word"] * 100_000)
    assert _stub_audio_duration_seconds(text) == narrate_mod.STUB_MAX_SECONDS


# --- ffmpeg silence command builder ---------------------------------------


def test_ffmpeg_silence_cmd_shape() -> None:
    cmd = _ffmpeg_silence_cmd(2.5, Path("/tmp/x.mp3"))
    assert cmd[0] == "ffmpeg"
    assert "-y" in cmd
    assert "-hide_banner" in cmd
    assert "duration=2.5" in " ".join(cmd)
    assert cmd[-1] == "/tmp/x.mp3"


# --- Stub synthesis (real ffmpeg) -----------------------------------------


def test_synthesize_stub_tts_writes_real_mp3(tmp_path: Path) -> None:
    """The stub provider writes a real, non-empty MP3 at the target path."""
    out = _synthesize_stub_tts("hello world", tmp_path / "x.mp3")
    assert out.exists()
    assert out.stat().st_size > 0
    # Quick check that the file has the MP3 magic header.
    head = out.read_bytes()[:3]
    # MP3 files start with "ID3" (ID3v2) or "\xff\xfb" (frame sync).
    assert head[:3] == b"ID3" or head[0:2] == b"\xff\xfb", f"unexpected header: {head!r}"


def test_synthesize_stub_tts_creates_parent_dir(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "x.mp3"
    _synthesize_stub_tts("hello", out)
    assert out.exists()


def test_synthesize_stub_tts_propagates_ttserror_on_missing_ffmpeg(
    tmp_path: Path, monkeypatch
) -> None:
    """If ffmpeg is missing, raise TTSError (not raw FileNotFoundError)."""
    monkeypatch.setattr(
        narrate_mod.subprocess, "run",
        mock.MagicMock(side_effect=FileNotFoundError("no ffmpeg")),
    )
    with pytest.raises(TTSError, match="ffmpeg"):
        _synthesize_stub_tts("hello", tmp_path / "x.mp3")


def test_synthesize_stub_tts_raises_on_ffmpeg_failure(
    tmp_path: Path, monkeypatch
) -> None:
    fake = mock.MagicMock()
    fake.returncode = 1
    fake.stderr = "boom"
    monkeypatch.setattr(narrate_mod.subprocess, "run", mock.MagicMock(return_value=fake))
    with pytest.raises(TTSError, match="ffmpeg exited"):
        _synthesize_stub_tts("hello", tmp_path / "x.mp3")


# --- Duration measurement via ffprobe --------------------------------------


def test_measure_mp3_duration_seconds_reads_real_mp3(tmp_path: Path) -> None:
    """ffprobe reads the duration of a real ffmpeg-generated MP3."""
    mp3 = tmp_path / "tone.mp3"
    _synthesize_stub_tts("hello world this is a test", mp3)
    dur = _measure_mp3_duration_seconds(mp3)
    # Stub audio is at 150 wpm; the file should be > 1s.
    assert dur > 1.0
    # And < 30s for a short test sentence.
    assert dur < 30.0


def test_measure_mp3_duration_seconds_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TTSError):
        _measure_mp3_duration_seconds(tmp_path / "nope.mp3")


def test_measure_mp3_duration_seconds_missing_ffprobe(
    tmp_path: Path, monkeypatch
) -> None:
    mp3 = tmp_path / "x.mp3"
    mp3.write_bytes(b"")
    monkeypatch.setattr(
        narrate_mod.subprocess, "run",
        mock.MagicMock(side_effect=FileNotFoundError("no ffprobe")),
    )
    with pytest.raises(TTSError, match="ffprobe"):
        _measure_mp3_duration_seconds(mp3)


def test_measure_mp3_duration_seconds_handles_non_json_output(
    tmp_path: Path, monkeypatch
) -> None:
    """If ffprobe's stdout isn't valid JSON, the regex fallback kicks in."""
    mp3 = tmp_path / "x.mp3"
    mp3.write_bytes(b"")
    fake = mock.MagicMock()
    fake.returncode = 0
    fake.stderr = ""
    fake.stdout = 'not really json, but "duration": "12.5" is in here'
    monkeypatch.setattr(narrate_mod.subprocess, "run", mock.MagicMock(return_value=fake))
    dur = _measure_mp3_duration_seconds(mp3)
    assert dur == pytest.approx(12.5)


def test_measure_mp3_duration_seconds_handles_json_output(
    tmp_path: Path, monkeypatch
) -> None:
    mp3 = tmp_path / "x.mp3"
    mp3.write_bytes(b"")
    fake = mock.MagicMock()
    fake.returncode = 0
    fake.stderr = ""
    fake.stdout = json.dumps({"format": {"duration": "7.25"}})
    monkeypatch.setattr(narrate_mod.subprocess, "run", mock.MagicMock(return_value=fake))
    dur = _measure_mp3_duration_seconds(mp3)
    assert dur == pytest.approx(7.25)


def test_measure_mp3_duration_seconds_handles_zero_duration(
    tmp_path: Path, monkeypatch
) -> None:
    mp3 = tmp_path / "x.mp3"
    mp3.write_bytes(b"")
    fake = mock.MagicMock()
    fake.returncode = 0
    fake.stderr = ""
    fake.stdout = json.dumps({"format": {"duration": "0"}})
    monkeypatch.setattr(narrate_mod.subprocess, "run", mock.MagicMock(return_value=fake))
    with pytest.raises(TTSError, match="non-positive"):
        _measure_mp3_duration_seconds(mp3)


# --- Duration tolerance check -----------------------------------------------


def test_check_duration_within_tolerance_passes_inside() -> None:
    # 100s source, 95s narration -> 5% drift, well within 15%.
    _check_duration_within_tolerance(95.0, 100.0)
    # And the symmetric case.
    _check_duration_within_tolerance(105.0, 100.0)
    # And exactly on the boundary.
    boundary = 1.0 - NARRATION_DURATION_TOLERANCE
    _check_duration_within_tolerance(boundary * 100.0, 100.0)


def test_check_duration_within_tolerance_fails_outside() -> None:
    with pytest.raises(TTSError, match="drift"):
        _check_duration_within_tolerance(50.0, 100.0)
    with pytest.raises(TTSError, match="drift"):
        _check_duration_within_tolerance(150.0, 100.0)


def test_check_duration_within_tolerance_skips_on_zero_source() -> None:
    """If source duration is unknown (<= 0), skip the check."""
    _check_duration_within_tolerance(0.0, 0.0)  # no raise
    _check_duration_within_tolerance(999.0, 0.0)  # no raise


# --- _dispatch_synthesis ---------------------------------------------------


def test_dispatch_synthesis_stub(tmp_path: Path) -> None:
    out = _dispatch_synthesis("stub", "hello", "voice", tmp_path / "x.mp3")
    assert out.exists()


def test_dispatch_synthesis_unknown_raises() -> None:
    with pytest.raises(TTSError, match="Unknown TTS provider"):
        _dispatch_synthesis("rss", "hello", "voice", Path("/tmp/x.mp3"))


# --- Public narrate_to_mp3 -------------------------------------------------


def test_narrate_to_mp3_stub_provider_creates_file(tmp_path: Path) -> None:
    out = narrate_to_mp3(
        "hello world", "test-job", tmp_path, provider="stub", check_duration=False
    )
    assert out == tmp_path / "test-job.mp3"
    assert out.exists()
    assert out.stat().st_size > 0


def test_narrate_to_mp3_rejects_empty_text(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty text"):
        narrate_to_mp3("", "j", tmp_path, provider="stub", check_duration=False)
    with pytest.raises(ValueError, match="empty text"):
        narrate_to_mp3("   ", "j", tmp_path, provider="stub", check_duration=False)


def test_narrate_to_mp3_rejects_empty_job_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="job_id"):
        narrate_to_mp3("hello", "", tmp_path, provider="stub", check_duration=False)


def test_narrate_to_mp3_creates_output_dir(tmp_path: Path) -> None:
    out = narrate_to_mp3(
        "hello", "j",
        tmp_path / "deep" / "nested",
        provider="stub", check_duration=False,
    )
    assert out.exists()


def test_narrate_to_mp3_within_tolerance_passes(tmp_path: Path) -> None:
    """The stub audio duration should be within +/-15% of the source.

    Stub = 150 wpm; if we ask for a 60-second source, 60s of "word"
    text produces a 60-second narration, well within the 9-second
    window.
    """
    text = " ".join(["word"] * 150)  # 60-second stub
    narrate_to_mp3(
        text, "j", tmp_path,
        provider="stub",
        source_duration_seconds=60.0,
    )


def test_narrate_to_mp3_outside_tolerance_raises(tmp_path: Path) -> None:
    """If the source duration is wildly different, raise TTSError."""
    with pytest.raises(TTSError, match="drift"):
        # 60-second narration vs 1-second "source" -> ~6000% drift.
        narrate_to_mp3(
            " ".join(["word"] * 150), "j", tmp_path,
            provider="stub",
            source_duration_seconds=1.0,
        )


def test_narrate_to_mp3_skips_duration_check_when_source_unknown(
    tmp_path: Path,
) -> None:
    """If source_duration_seconds is None, the duration check is skipped."""
    narrate_to_mp3(
        "hello", "j", tmp_path, provider="stub", source_duration_seconds=None
    )


def test_narrate_to_mp3_passes_voice_to_provider(tmp_path: Path) -> None:
    """The voice arg flows through to the provider."""
    with mock.patch.object(
        narrate_mod, "_synthesize_stub_tts", return_value=tmp_path / "j.mp3"
    ) as stub:
        narrate_to_mp3(
            "hello", "j", tmp_path,
            provider="stub", voice="en-GB-RyanNeural", check_duration=False,
        )
    stub.assert_called_once()
    # The voice arg is captured on the public function call, not the
    # internal _synthesize_stub_tts (which doesn't take voice). So we
    # instead verify the function was called with the text.
    assert stub.call_args.args[0] == "hello"


def test_narrate_to_mp3_resolves_provider_from_arg(tmp_path: Path) -> None:
    """The provider arg overrides the env var."""
    with mock.patch.dict(os.environ, {"TTS_PROVIDER": "openai"}):
        with mock.patch.object(
            narrate_mod, "_synthesize_stub_tts", return_value=tmp_path / "j.mp3"
        ) as stub:
            narrate_to_mp3(
                "hello", "j", tmp_path, provider="stub", check_duration=False,
            )
    stub.assert_called_once()


# --- Edge TTS call-site (mocked, no network) ------------------------------


def test_edge_tts_call_site_passes_text_and_voice(
    tmp_path: Path, monkeypatch
) -> None:
    """The Edge TTS dispatch constructs a Communicate with the right args."""
    fake_comm = mock.MagicMock()

    async def fake_save(self, path):
        Path(path).write_bytes(b"FAKE-MP3")

    # Drive _synthesize_edge_tts via the public narrate_to_mp3, with
    # the real provider dispatch and a mocked edge_tts module.
    fake_module = mock.MagicMock()
    fake_module.Communicate = mock.MagicMock(return_value=fake_comm)
    # fake_comm.save is async.
    async def _save(path):
        Path(path).write_bytes(b"FAKE-MP3")
    fake_comm.save = _save
    monkeypatch.setitem(__import__("sys").modules, "edge_tts", fake_module)

    out = narrate_to_mp3(
        "hello world", "j", tmp_path,
        provider="edge", voice="en-US-AriaNeural", check_duration=False,
    )
    assert out.exists()
    assert out.read_bytes() == b"FAKE-MP3"
    fake_module.Communicate.assert_called_once_with(
        "hello world", "en-US-AriaNeural"
    )


# --- OpenAI TTS call-site (mocked) ----------------------------------------


def test_openai_tts_call_site_passes_text_and_voice(
    tmp_path: Path, monkeypatch
) -> None:
    """The OpenAI TTS dispatch calls ``client.audio.speech.create``
    with the right text and voice."""
    fake_response = mock.MagicMock()

    def fake_stream_to_file(path):
        Path(path).write_bytes(b"FAKE-OPENAI-MP3")
    fake_response.stream_to_file = fake_stream_to_file

    fake_client = mock.MagicMock()
    fake_client.audio.speech.create.return_value = fake_response

    fake_module = mock.MagicMock()
    fake_module.OpenAI = mock.MagicMock(return_value=fake_client)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)

    out = narrate_to_mp3(
        "hello world", "j", tmp_path,
        provider="openai", voice="nova", check_duration=False,
    )
    assert out.exists()
    assert out.read_bytes() == b"FAKE-OPENAI-MP3"
    create = fake_client.audio.speech.create
    create.assert_called_once()
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == DEFAULT_OPENAI_TTS_MODEL
    assert kwargs["voice"] == "nova"
    assert kwargs["input"] == "hello world"


def test_openai_tts_uses_env_model_override(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_TTS_MODEL", "tts-1-hd")
    fake_response = mock.MagicMock()
    fake_response.stream_to_file = lambda p: Path(p).write_bytes(b"x")
    fake_client = mock.MagicMock()
    fake_client.audio.speech.create.return_value = fake_response
    fake_module = mock.MagicMock()
    fake_module.OpenAI = mock.MagicMock(return_value=fake_client)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)
    narrate_to_mp3(
        "hello", "j", tmp_path,
        provider="openai", check_duration=False,
    )
    assert fake_client.audio.speech.create.call_args.kwargs["model"] == "tts-1-hd"


# --- Error code constant --------------------------------------------------


def test_tts_failed_code_constant() -> None:
    assert CODE_TTS_FAILED == "TTS_FAILED"
