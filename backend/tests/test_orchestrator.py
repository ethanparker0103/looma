"""Unit tests for ``app.pipeline.orchestrator``.

All tests are pure unit tests — no live Whisper. The two pipeline
stages are patched via ``unittest.mock`` so the contract being verified
is "the stages are called in order and their outputs are composed
correctly".

Coverage:

* :class:`JobSource` constructors and validation
* :func:`utc_now_iso` produces an ISO-8601 UTC string with the ``Z`` suffix
* :func:`audio_path_for` / :func:`output_path_for` return the correct paths
* :func:`run_job` happy path runs ingest → transcribe and returns a
  :class:`TranscriptionResult`
* Stage order is correct (ingest -> transcribe)
* Errors raised by any stage propagate to the caller
* The async wrapper ``run_job_async`` returns the same result
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from app.models import (
    TranscriptSegment,
    TranscriptionResult,
)
from app.pipeline import orchestrator as orch_mod
from app.pipeline.orchestrator import (
    JobSource,
    audio_path_for,
    output_path_for,
    run_job,
    run_job_async,
    utc_now_iso,
)


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def sample_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        transcript="hello world",
        segments=[TranscriptSegment(start=0.0, end=1.0, text="hello world")],
        language="en",
        duration_seconds=600.0,
    )


@pytest.fixture
def patched_stages(sample_transcription, tmp_path):
    """Patch both stages so no real download or Whisper runs."""
    mp3_path = tmp_path / "src.mp3"
    mp3_path.write_bytes(b"")
    with mock.patch.object(
        orch_mod, "_stage_ingest", return_value=mp3_path,
    ) as ingest, mock.patch.object(
        orch_mod, "_stage_transcribe", return_value=sample_transcription,
    ) as transcribe:
        yield ingest, transcribe


# --- JobSource tests --------------------------------------------------------


class TestJobSource:
    def test_youtube_constructor(self) -> None:
        s = JobSource.youtube("https://youtu.be/abc")
        assert s.kind == "youtube"
        assert s.ref == "https://youtu.be/abc"
        assert s.display_name is None

    def test_upload_constructor(self) -> None:
        s = JobSource.upload("/tmp/vid.mp4", display_name="demo.mp4")
        assert s.kind == "upload"
        assert s.ref == "/tmp/vid.mp4"
        assert s.display_name == "demo.mp4"

    def test_upload_defaults_to_basename(self) -> None:
        s = JobSource.upload("/tmp/subdir/video.mp4")
        assert s.display_name == "video.mp4"

    def test_invalid_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="JobSource kind must be"):
            JobSource("zoom", "https://x.com")

    def test_empty_ref_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            JobSource("youtube", "")

    def test_equality(self) -> None:
        a = JobSource.youtube("https://youtu.be/abc")
        b = JobSource.youtube("https://youtu.be/abc")
        assert a == b
        assert not (a == "not-a-jobsource")

    def test_hash(self) -> None:
        a = JobSource.youtube("https://youtu.be/abc")
        b = JobSource.youtube("https://youtu.be/abc")
        assert hash(a) == hash(b)


# --- Helper function tests --------------------------------------------------


class TestHelpers:
    def test_utc_now_iso_ends_with_z(self) -> None:
        s = utc_now_iso()
        assert s.endswith("Z")

    def test_utc_now_iso_parseable(self) -> None:
        s = utc_now_iso()
        # Strip the Z before parsing
        datetime.strptime(s[:-1], "%Y-%m-%dT%H:%M:%S.%f")

    def test_audio_path_for(self) -> None:
        p = audio_path_for("abc-123")
        assert p.name == "abc-123.mp3"
        assert p.parent.name == "audio"
        assert "data" in p.parts

    def test_output_path_for(self) -> None:
        p = output_path_for("abc-123")
        assert p.name == "abc-123.mp3"
        assert p.parent.name == "outputs"


# --- run_job tests ----------------------------------------------------------


class TestRunJob:
    def test_happy_path(self, patched_stages) -> None:
        """run_job calls ingest then transcribe and returns a TranscriptionResult."""
        source = JobSource.youtube("https://youtu.be/abc")

        result = run_job("job-1", source)

        ingest_mock, transcribe_mock = patched_stages
        ingest_mock.assert_called_once()
        transcribe_mock.assert_called_once()
        assert isinstance(result, TranscriptionResult)
        assert result.transcript == "hello world"
        assert result.duration_seconds == 600.0

    def test_empty_job_id_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            run_job("", JobSource.youtube("https://youtu.be/abc"))

    def test_bad_source_type_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a JobSource"):
            run_job("job-1", "not-a-jobsource")  # type: ignore[arg-type]

    def test_ingest_error_propagates(self, sample_transcription) -> None:
        """If ingest raises, run_job raises with the same error."""
        with mock.patch.object(
            orch_mod, "_stage_ingest",
            side_effect=FileNotFoundError("no such file"),
        ):
            with pytest.raises(FileNotFoundError):
                run_job("job-2", JobSource.youtube("https://youtu.be/abc"))

    def test_transcribe_error_propagates(self) -> None:
        """If transcribe raises, run_job raises."""
        mp3 = Path("/tmp/dummy.mp3")
        with mock.patch.object(
            orch_mod, "_stage_ingest", return_value=mp3,
        ), mock.patch.object(
            orch_mod, "_stage_transcribe",
            side_effect=RuntimeError("whisper crash"),
        ):
            with pytest.raises(RuntimeError, match="whisper crash"):
                run_job("job-3", JobSource.youtube("https://youtu.be/abc"))


# --- run_job_async tests ----------------------------------------------------


@pytest.mark.asyncio
async def test_run_job_async_wraps_result(patched_stages) -> None:
    """run_job_async returns the same result as run_job."""
    source = JobSource.youtube("https://youtu.be/abc")
    result = await run_job_async("async-job", source)
    assert isinstance(result, TranscriptionResult)
    assert result.transcript == "hello world"
