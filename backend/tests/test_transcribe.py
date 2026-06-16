"""Unit tests for ``app.pipeline.transcribe`` (AC-4).

All tests are pure unit tests — no live Whisper model load, no audio
processing. ``whisper.load_model`` and the model object's
``transcribe`` method are patched via ``unittest.mock`` so the
contract being verified is "Whisper's output is mapped correctly to
``TranscriptionResult`` and the right exceptions are raised".
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from app.pipeline import transcribe as transcribe_mod
from app.pipeline.transcribe import (
    DEFAULT_LANGUAGE,
    DEFAULT_WHISPER_MODEL,
    TranscriptionError,
    _load_model,
    _resolve_model_name,
    transcribe,
)


# --- Helpers ---------------------------------------------------------------


@pytest.fixture
def fake_mp3(tmp_path: Path) -> Path:
    """Create a real (empty) MP3 placeholder so the existence check passes.

    The transcribe layer only needs the file to exist; the contents are
    irrelevant because the model is mocked.
    """
    p = tmp_path / "audio.mp3"
    p.write_bytes(b"")
    return p


def _fake_whisper_result(
    *,
    text: str = "Hello world.",
    segments: list[dict] | None = None,
    language: str = "en",
) -> dict:
    """Build a dict shaped like ``whisper.Whisper.transcribe``'s return."""
    if segments is None:
        segments = [
            {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello "},
            {"id": 1, "start": 1.5, "end": 3.0, "text": "world."},
        ]
    return {"text": text, "segments": segments, "language": language}


def _make_model_mock(*, result: dict) -> mock.MagicMock:
    """Build a mock Whisper model whose ``transcribe`` returns ``result``."""
    model = mock.MagicMock()
    model.transcribe.return_value = result
    return model


def _patch_load_model(model: mock.MagicMock):
    """Patch ``whisper.load_model`` to return ``model`` and clear the cache."""
    transcribe_mod._load_model.cache_clear()
    return mock.patch.object(transcribe_mod.whisper, "load_model", return_value=model)


# --- transcribe happy path ------------------------------------------------


def test_transcribe_returns_typed_result(fake_mp3: Path) -> None:
    fake = _fake_whisper_result()
    with _patch_load_model(_make_model_mock(result=fake)):
        out = transcribe(fake_mp3, model_name="tiny")

    assert out.transcript == "Hello world."
    assert out.language == "en"
    assert out.duration_seconds == pytest.approx(3.0)
    assert len(out.segments) == 2
    assert out.segments[0].start == 0.0
    assert out.segments[0].end == 1.5
    assert out.segments[0].text == "Hello"
    assert out.segments[1].end == 3.0


def test_transcribe_invokes_model_with_path_string(fake_mp3: Path) -> None:
    fake = _fake_whisper_result()
    model = _make_model_mock(result=fake)
    with _patch_load_model(model):
        transcribe(fake_mp3, model_name="tiny")

    model.transcribe.assert_called_once()
    (called_arg,) = model.transcribe.call_args.args
    # Whisper's API takes a string path; we always pass a string.
    assert isinstance(called_arg, str)
    assert called_arg == str(fake_mp3)


def test_transcribe_accepts_string_path(tmp_path: Path) -> None:
    p = tmp_path / "audio.mp3"
    p.write_bytes(b"")
    fake = _fake_whisper_result()
    with _patch_load_model(_make_model_mock(result=fake)):
        out = transcribe(str(p), model_name="tiny")
    assert out.transcript == "Hello world."


def test_transcribe_uses_configured_model_name(fake_mp3: Path) -> None:
    fake = _fake_whisper_result()
    with _patch_load_model(_make_model_mock(result=fake)) as load:
        transcribe(fake_mp3, model_name="medium")
    load.assert_called_once_with("medium")


# --- model-name resolution ------------------------------------------------


def test_resolve_model_name_prefers_explicit_arg() -> None:
    assert _resolve_model_name("large") == "large"


def test_resolve_model_name_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("WHISPER_MODEL", "base")
    assert _resolve_model_name(None) == "base"


def test_resolve_model_name_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    assert _resolve_model_name(None) == DEFAULT_WHISPER_MODEL == "small"


def test_transcribe_uses_env_when_arg_omitted(
    fake_mp3: Path, monkeypatch
) -> None:
    monkeypatch.setenv("WHISPER_MODEL", "base")
    fake = _fake_whisper_result()
    with _patch_load_model(_make_model_mock(result=fake)) as load:
        transcribe(fake_mp3)
    load.assert_called_once_with("base")


# --- segments handling ---------------------------------------------------


def test_transcribe_maps_all_segment_fields(fake_mp3: Path) -> None:
    fake = _fake_whisper_result(
        segments=[
            {"start": 10.0, "end": 12.5, "text": " first "},
            {"start": 12.5, "end": 15.0, "text": " second"},
        ]
    )
    with _patch_load_model(_make_model_mock(result=fake)):
        out = transcribe(fake_mp3, model_name="tiny")
    assert [s.text for s in out.segments] == ["first", "second"]


def test_transcribe_sorts_segments_by_start(fake_mp3: Path) -> None:
    # Whisper guarantees order, but we don't trust third parties.
    fake = _fake_whisper_result(
        segments=[
            {"start": 5.0, "end": 6.0, "text": "b"},
            {"start": 1.0, "end": 2.0, "text": "a"},
        ]
    )
    with _patch_load_model(_make_model_mock(result=fake)):
        out = transcribe(fake_mp3, model_name="tiny")
    assert [s.text for s in out.segments] == ["a", "b"]


def test_transcribe_handles_empty_segments(fake_mp3: Path) -> None:
    fake = _fake_whisper_result(text="", segments=[])
    with _patch_load_model(_make_model_mock(result=fake)):
        out = transcribe(fake_mp3, model_name="tiny")
    assert out.transcript == ""
    assert out.segments == []
    assert out.duration_seconds == 0.0


def test_transcribe_duration_uses_last_segment_end(fake_mp3: Path) -> None:
    fake = _fake_whisper_result(
        segments=[
            {"start": 0.0, "end": 4.0, "text": "a"},
            {"start": 4.0, "end": 9.5, "text": "b"},
            {"start": 9.5, "end": 12.0, "text": "c"},
        ]
    )
    with _patch_load_model(_make_model_mock(result=fake)):
        out = transcribe(fake_mp3, model_name="tiny")
    assert out.duration_seconds == pytest.approx(12.0)


# --- language & transcript fields ----------------------------------------


def test_transcribe_defaults_language_when_missing(fake_mp3: Path) -> None:
    raw = {"text": "hi", "segments": []}  # no "language" key
    with _patch_load_model(_make_model_mock(result=raw)):
        out = transcribe(fake_mp3, model_name="tiny")
    assert out.language == DEFAULT_LANGUAGE == "en"


def test_transcribe_defaults_language_when_empty(fake_mp3: Path) -> None:
    raw = {"text": "hi", "segments": [], "language": ""}
    with _patch_load_model(_make_model_mock(result=raw)):
        out = transcribe(fake_mp3, model_name="tiny")
    assert out.language == "en"


def test_transcribe_handles_missing_text(fake_mp3: Path) -> None:
    raw = {"segments": [{"start": 0.0, "end": 1.0, "text": "x"}], "language": "en"}
    with _patch_load_model(_make_model_mock(result=raw)):
        out = transcribe(fake_mp3, model_name="tiny")
    assert out.transcript == ""


# --- error handling ------------------------------------------------------


def test_transcribe_raises_when_mp3_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        transcribe(tmp_path / "does_not_exist.mp3", model_name="tiny")


def test_transcribe_wraps_whisper_exception(fake_mp3: Path) -> None:
    model = mock.MagicMock()
    model.transcribe.side_effect = RuntimeError("model exploded")
    with _patch_load_model(model):
        with pytest.raises(TranscriptionError) as excinfo:
            transcribe(fake_mp3, model_name="tiny")
    assert "model exploded" in str(excinfo.value)


# --- model cache --------------------------------------------------------


def test_load_model_calls_whisper_with_name() -> None:
    transcribe_mod._load_model.cache_clear()
    with mock.patch.object(
        transcribe_mod.whisper, "load_model", return_value=mock.MagicMock()
    ) as load:
        m1 = _load_model("small")
        m2 = _load_model("small")
    load.assert_called_once_with("small")
    assert m1 is m2  # cached


def test_load_model_caches_per_name() -> None:
    transcribe_mod._load_model.cache_clear()
    with mock.patch.object(
        transcribe_mod.whisper, "load_model", return_value=mock.MagicMock()
    ) as load:
        _load_model("tiny")
        _load_model("small")
        _load_model("tiny")  # cache hit
    assert load.call_count == 2
    assert load.call_args_list[0].args == ("tiny",)
    assert load.call_args_list[1].args == ("small",)


# --- end-to-end smoke with all fields ------------------------------------


def test_transcribe_full_result_shape(fake_mp3: Path) -> None:
    """Smoke test: result has every field required by AC-4."""
    fake = _fake_whisper_result(
        text="  The quick brown fox.  ",
        segments=[{"start": 0.0, "end": 4.2, "text": "The quick brown fox."}],
        language="en",
    )
    with _patch_load_model(_make_model_mock(result=fake)):
        out = transcribe(fake_mp3, model_name="tiny")

    # All AC-4 required fields:
    assert hasattr(out, "transcript")
    assert hasattr(out, "segments")
    assert hasattr(out, "language")
    assert hasattr(out, "duration_seconds")
    # Per-segment required fields:
    for s in out.segments:
        assert hasattr(s, "start")
        assert hasattr(s, "end")
        assert hasattr(s, "text")
    # Transcript is stripped; duration is positive; language is set.
    assert out.transcript == "The quick brown fox."
    assert out.duration_seconds == pytest.approx(4.2)
    assert out.language == "en"
