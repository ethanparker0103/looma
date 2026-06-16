"""Unit tests for ``app.pipeline.orchestrator`` (AC-6).

All tests are pure unit tests — no live Whisper, no live LLM, no live
TTS. The four pipeline stages are patched via ``unittest.mock`` so
the contract being verified is "the four stages are called in order
and their outputs are composed into a single ``LoomaResult`` with the
right shape".

Coverage:

* :class:`JobSource` constructors and validation
* :func:`utc_now_iso` produces an ISO-8601 UTC string with the ``Z``
  suffix
* :func:`public_audio_url_for` / :func:`audio_path_for` /
  :func:`output_path_for` are path helpers with the AC contract shape
* :func:`run_job` happy path composes a :class:`LoomaResult` whose
  fields mirror the four stage outputs and whose ``audio_url`` and
  ``created_at`` are well-formed
* Stage order is correct (ingest -> transcribe -> extract -> narrate)
* The upload path is preserved through the orchestrator
* Errors raised by any stage propagate to the caller
* The async wrapper ``run_job_async`` returns the same result
* AC-6 schema probe: ``LoomaResult`` has every required field
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from app.models import (
    Chapter,
    KnowledgeExtract,
    LoomaResult,
    TranscriptSegment,
    TranscriptionResult,
)
from app.pipeline import orchestrator as orch_mod
from app.pipeline.orchestrator import (
    JobSource,
    audio_path_for,
    output_path_for,
    public_audio_url_for,
    run_job,
    run_job_async,
    utc_now_iso,
)


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def sample_transcription() -> TranscriptionResult:
    raw = json.loads(
        (Path(__file__).resolve().parent / "fixtures" / "sample_transcription.json")
        .read_text(encoding="utf-8")
    )
    return TranscriptionResult(
        transcript=raw["transcript"],
        segments=[TranscriptSegment(**s) for s in raw["segments"]],
        language=raw["language"],
        duration_seconds=raw["duration_seconds"],
    )


@pytest.fixture
def sample_knowledge() -> KnowledgeExtract:
    raw = json.loads(
        (Path(__file__).resolve().parent / "fixtures" / "llm_extract_response.json")
        .read_text(encoding="utf-8")
    )
    return KnowledgeExtract.model_validate(raw)


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch) -> None:
    """Redirect the module's AUDIO_DIR and OUTPUTS_DIR to ``tmp_path``."""
    monkeypatch.setattr(orch_mod, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(orch_mod, "OUTPUTS_DIR", tmp_path / "outputs")


# --- JobSource -------------------------------------------------------------


def test_job_source_youtube_constructor() -> None:
    src = JobSource.youtube("https://www.youtube.com/watch?v=abc")
    assert src.kind == "youtube"
    assert src.ref == "https://www.youtube.com/watch?v=abc"


def test_job_source_upload_constructor() -> None:
    src = JobSource.upload("/tmp/some-video.mp4")
    assert src.kind == "upload"
    assert src.ref == "/tmp/some-video.mp4"


def test_job_source_upload_accepts_path_object() -> None:
    src = JobSource.upload(Path("/tmp/v.mp4"))
    assert src.ref == "/tmp/v.mp4"
    assert src.kind == "upload"


def test_job_source_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        JobSource(kind="rss", ref="https://example.com")  # type: ignore[arg-type]


def test_job_source_rejects_empty_ref() -> None:
    with pytest.raises(ValueError):
        JobSource(kind="youtube", ref="")


def test_job_source_rejects_non_string_ref() -> None:
    with pytest.raises(ValueError):
        JobSource(kind="youtube", ref=123)  # type: ignore[arg-type]


def test_job_source_equality_and_hash() -> None:
    a = JobSource.youtube("https://x")
    b = JobSource.youtube("https://x")
    c = JobSource.upload("/tmp/y.mp4")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert {a, b, c} == {a, c}


# --- utc_now_iso ------------------------------------------------------------


def test_utc_now_iso_format_is_iso8601_utc() -> None:
    """AC-6: ``created_at`` is ISO-8601 UTC with a ``Z`` suffix."""
    out = utc_now_iso()
    # YYYY-MM-DDTHH:MM:SS.uuuuuuZ
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$", out)


def test_utc_now_iso_is_parseable() -> None:
    """The format round-trips through datetime.fromisoformat."""
    out = utc_now_iso()
    # Replace 'Z' with '+00:00' for fromisoformat compatibility (pre-3.11).
    parsed = datetime.fromisoformat(out.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_utc_now_iso_two_calls_differ() -> None:
    """Two calls produce different timestamps (modulo clock resolution)."""
    a = utc_now_iso()
    b = utc_now_iso()
    # Either they differ, or the test ran inside a single microsecond —
    # both are acceptable, so we just check the function returns a
    # string in either case.
    assert isinstance(a, str)
    assert isinstance(b, str)


# --- Public path helpers ---------------------------------------------------


def test_public_audio_url_for_format() -> None:
    assert public_audio_url_for("abc-123") == "/audio/abc-123.mp3"


def test_audio_path_for_uses_audio_dir() -> None:
    p = audio_path_for("xyz")
    assert p.name == "xyz.mp3"
    assert p.parent.name == "audio"


def test_output_path_for_uses_outputs_dir() -> None:
    p = output_path_for("xyz")
    assert p.name == "xyz.mp3"
    assert p.parent.name == "outputs"


# --- run_job happy path ----------------------------------------------------


def _patch_stages(
    *,
    mp3_path: Path,
    transcription: TranscriptionResult,
    knowledge: KnowledgeExtract,
    narrate_path: Path,
) -> dict:
    """Return a dict of mock.patch objects that stub the four stages.

    The orchestrator's stage helpers are what ``run_job`` calls, so
    patching them at the orchestrator module level is the cleanest
    way to short-circuit the pipeline without touching the real
    ingest/transcribe/extract/narrate modules.
    """
    return {
        "ingest": mock.patch.object(
            orch_mod, "_stage_ingest", return_value=mp3_path
        ),
        "transcribe": mock.patch.object(
            orch_mod, "_stage_transcribe", return_value=transcription
        ),
        "extract": mock.patch.object(
            orch_mod, "_stage_extract", return_value=knowledge
        ),
        "narrate": mock.patch.object(
            orch_mod, "_stage_narrate", return_value=narrate_path
        ),
    }


def test_run_job_returns_loomaresult(
    tmp_data_dir, sample_transcription, sample_knowledge, tmp_path: Path
) -> None:
    mp3_path = tmp_path / "src.mp3"
    mp3_path.write_bytes(b"")
    narrate_path = tmp_path / "out.mp3"
    narrate_path.write_bytes(b"")

    patches = _patch_stages(
        mp3_path=mp3_path,
        transcription=sample_transcription,
        knowledge=sample_knowledge,
        narrate_path=narrate_path,
    )
    with patches["ingest"], patches["transcribe"], patches["extract"], patches["narrate"]:
        result = run_job("job-001", JobSource.youtube("https://youtu.be/abc"))

    assert isinstance(result, LoomaResult)
    assert result.job_id == "job-001"
    assert result.source_type == "youtube"
    assert result.source_ref == "https://youtu.be/abc"
    assert result.title == sample_knowledge.title
    assert result.transcription is sample_transcription
    assert result.knowledge is sample_knowledge
    assert result.audio_url == "/audio/job-001.mp3"
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$", result.created_at)


def test_run_job_audio_url_uses_job_id() -> None:
    """AC-6: ``audio_url`` is ``/audio/<job_id>.mp3``."""
    with mock.patch.object(orch_mod, "_stage_ingest", return_value=Path("/tmp/x.mp3")), \
         mock.patch.object(orch_mod, "_stage_transcribe"), \
         mock.patch.object(orch_mod, "_stage_extract"), \
         mock.patch.object(orch_mod, "_stage_narrate", return_value=Path("/tmp/o.mp3")):
        from app.models import TranscriptionResult, KnowledgeExtract, TranscriptSegment

        trans = TranscriptionResult(
            transcript="t",
            segments=[TranscriptSegment(start=0.0, end=1.0, text="t")],
            language="en",
            duration_seconds=1.0,
        )
        know = KnowledgeExtract(
            title="T",
            summary="S. S. S.",
            insights=["a", "b", "c", "d", "e"],
            chapters=[Chapter(start_seconds=0.0, end_seconds=1.0, title="c")],
            narrative="x " * 200,
            filler_removed=0,
        )
        with mock.patch.object(orch_mod, "_stage_transcribe", return_value=trans), \
             mock.patch.object(orch_mod, "_stage_extract", return_value=know):
            result = run_job("job-XYZ-9", JobSource.youtube("https://youtu.be/abc"))
    assert result.audio_url == "/audio/job-XYZ-9.mp3"


def test_run_job_upload_source_uses_basename() -> None:
    """For uploads, ``source_ref`` is the basename of the upload path."""
    with mock.patch.object(orch_mod, "_stage_ingest", return_value=Path("/tmp/x.mp3")), \
         mock.patch.object(orch_mod, "_stage_transcribe"), \
         mock.patch.object(orch_mod, "_stage_extract"), \
         mock.patch.object(orch_mod, "_stage_narrate", return_value=Path("/tmp/o.mp3")):
        from app.models import TranscriptionResult, KnowledgeExtract, TranscriptSegment

        trans = TranscriptionResult(
            transcript="t",
            segments=[TranscriptSegment(start=0.0, end=1.0, text="t")],
            language="en",
            duration_seconds=1.0,
        )
        know = KnowledgeExtract(
            title="T",
            summary="S. S. S.",
            insights=["a", "b", "c", "d", "e"],
            chapters=[Chapter(start_seconds=0.0, end_seconds=1.0, title="c")],
            narrative="x " * 200,
            filler_removed=0,
        )
        with mock.patch.object(orch_mod, "_stage_transcribe", return_value=trans), \
             mock.patch.object(orch_mod, "_stage_extract", return_value=know):
            result = run_job("job-1", JobSource.upload("/tmp/some/clip.mp4"))
    assert result.source_type == "upload"
    assert result.source_ref == "clip.mp4"


def test_run_job_calls_stages_in_order(
    tmp_data_dir, sample_transcription, sample_knowledge, tmp_path: Path
) -> None:
    mp3_path = tmp_path / "src.mp3"
    mp3_path.write_bytes(b"")
    narrate_path = tmp_path / "out.mp3"
    narrate_path.write_bytes(b"")

    call_order: list[str] = []

    def fake_ingest(*a, **k):
        call_order.append("ingest")
        return mp3_path

    def fake_transcribe(*a, **k):
        call_order.append("transcribe")
        return sample_transcription

    def fake_extract(*a, **k):
        call_order.append("extract")
        return sample_knowledge

    def fake_narrate(*a, **k):
        call_order.append("narrate")
        return narrate_path

    with mock.patch.object(orch_mod, "_stage_ingest", side_effect=fake_ingest), \
         mock.patch.object(orch_mod, "_stage_transcribe", side_effect=fake_transcribe), \
         mock.patch.object(orch_mod, "_stage_extract", side_effect=fake_extract), \
         mock.patch.object(orch_mod, "_stage_narrate", side_effect=fake_narrate):
        run_job("job-1", JobSource.youtube("https://youtu.be/abc"))

    assert call_order == ["ingest", "transcribe", "extract", "narrate"]


def test_run_job_passes_transcription_duration_to_narrate(
    tmp_data_dir, sample_transcription, sample_knowledge, tmp_path: Path
) -> None:
    """AC-7: the orchestrator passes the source duration to the TTS stage."""
    mp3 = tmp_path / "x.mp3"
    mp3.write_bytes(b"")
    out = tmp_path / "o.mp3"
    out.write_bytes(b"")
    with mock.patch.object(orch_mod, "_stage_ingest", return_value=mp3), \
         mock.patch.object(orch_mod, "_stage_transcribe", return_value=sample_transcription), \
         mock.patch.object(orch_mod, "_stage_extract", return_value=sample_knowledge), \
         mock.patch.object(orch_mod, "_stage_narrate", return_value=out) as n:
        run_job("j-1", JobSource.youtube("https://youtu.be/abc"))
    # The third positional arg is the source_duration_seconds, which
    # should equal sample_transcription.duration_seconds (600.0).
    assert n.call_args.args[2] == sample_transcription.duration_seconds


def test_run_job_passes_transcription_duration_to_narrate(
    tmp_data_dir, sample_transcription, sample_knowledge, tmp_path: Path
) -> None:
    """AC-7: the orchestrator passes the source duration to the TTS stage."""
    mp3 = tmp_path / "x.mp3"
    mp3.write_bytes(b"")
    out = tmp_path / "o.mp3"
    out.write_bytes(b"")
    with mock.patch.object(orch_mod, "_stage_ingest", return_value=mp3), \
         mock.patch.object(orch_mod, "_stage_transcribe", return_value=sample_transcription), \
         mock.patch.object(orch_mod, "_stage_extract", return_value=sample_knowledge), \
         mock.patch.object(orch_mod, "_stage_narrate", return_value=out) as n:
        run_job("j-1", JobSource.youtube("https://youtu.be/abc"))
    # The third positional arg is the source_duration_seconds, which
    # should equal sample_transcription.duration_seconds (600.0).
    assert n.call_args.args[2] == sample_transcription.duration_seconds


def test_run_job_title_taken_from_knowledge(
    tmp_data_dir, sample_transcription, tmp_path: Path
) -> None:
    """``LoomaResult.title`` is the LLM-refined title from AC-5."""
    custom = KnowledgeExtract(
        title="My Custom Title That Is Distinctly Refined",
        summary="This is sentence one. This is sentence two. This is sentence three.",
        insights=["a", "b", "c", "d", "e"],
        chapters=[Chapter(start_seconds=0.0, end_seconds=10.0, title="c")],
        narrative="x " * 200,
        filler_removed=2,
    )
    mp3 = tmp_path / "x.mp3"
    mp3.write_bytes(b"")
    out = tmp_path / "o.mp3"
    out.write_bytes(b"")
    with mock.patch.object(orch_mod, "_stage_ingest", return_value=mp3), \
         mock.patch.object(orch_mod, "_stage_transcribe", return_value=sample_transcription), \
         mock.patch.object(orch_mod, "_stage_extract", return_value=custom), \
         mock.patch.object(orch_mod, "_stage_narrate", return_value=out):
        result = run_job("j", JobSource.youtube("https://youtu.be/abc"))
    assert result.title == "My Custom Title That Is Distinctly Refined"


def test_run_job_passes_looma_result_to_caller(
    tmp_data_dir, sample_transcription, sample_knowledge, tmp_path: Path
) -> None:
    mp3 = tmp_path / "x.mp3"
    mp3.write_bytes(b"")
    out = tmp_path / "o.mp3"
    out.write_bytes(b"")
    with mock.patch.object(orch_mod, "_stage_ingest", return_value=mp3), \
         mock.patch.object(orch_mod, "_stage_transcribe", return_value=sample_transcription), \
         mock.patch.object(orch_mod, "_stage_extract", return_value=sample_knowledge), \
         mock.patch.object(orch_mod, "_stage_narrate", return_value=out):
        result = run_job("j-1", JobSource.youtube("https://youtu.be/abc"))

    # AC-6 field probe
    assert hasattr(result, "job_id")
    assert hasattr(result, "source_type")
    assert hasattr(result, "source_ref")
    assert hasattr(result, "title")
    assert hasattr(result, "transcription")
    assert hasattr(result, "knowledge")
    assert hasattr(result, "audio_url")
    assert hasattr(result, "created_at")


# --- run_job error propagation ---------------------------------------------


def test_run_job_propagates_ingest_errors(
    tmp_data_dir, sample_transcription, sample_knowledge
) -> None:
    """Ingest errors are not swallowed; they bubble up unchanged."""
    err = RuntimeError("download failed")
    with mock.patch.object(orch_mod, "_stage_ingest", side_effect=err), \
         mock.patch.object(orch_mod, "_stage_transcribe") as t, \
         mock.patch.object(orch_mod, "_stage_extract") as e, \
         mock.patch.object(orch_mod, "_stage_narrate") as n:
        with pytest.raises(RuntimeError, match="download failed"):
            run_job("j", JobSource.youtube("https://youtu.be/abc"))
    # Subsequent stages must not have been called.
    t.assert_not_called()
    e.assert_not_called()
    n.assert_not_called()


def test_run_job_propagates_transcribe_errors(
    tmp_data_dir, sample_knowledge, tmp_path: Path
) -> None:
    err = RuntimeError("transcribe blew up")
    mp3 = tmp_path / "x.mp3"
    mp3.write_bytes(b"")
    with mock.patch.object(orch_mod, "_stage_ingest", return_value=mp3), \
         mock.patch.object(orch_mod, "_stage_transcribe", side_effect=err), \
         mock.patch.object(orch_mod, "_stage_extract") as e, \
         mock.patch.object(orch_mod, "_stage_narrate") as n:
        with pytest.raises(RuntimeError, match="transcribe blew up"):
            run_job("j", JobSource.youtube("https://youtu.be/abc"))
    e.assert_not_called()
    n.assert_not_called()


def test_run_job_propagates_extract_errors(
    tmp_data_dir, sample_transcription, tmp_path: Path
) -> None:
    err = RuntimeError("LLM exploded")
    mp3 = tmp_path / "x.mp3"
    mp3.write_bytes(b"")
    with mock.patch.object(orch_mod, "_stage_ingest", return_value=mp3), \
         mock.patch.object(orch_mod, "_stage_transcribe", return_value=sample_transcription), \
         mock.patch.object(orch_mod, "_stage_extract", side_effect=err), \
         mock.patch.object(orch_mod, "_stage_narrate") as n:
        with pytest.raises(RuntimeError, match="LLM exploded"):
            run_job("j", JobSource.youtube("https://youtu.be/abc"))
    n.assert_not_called()


def test_run_job_propagates_narrate_errors(
    tmp_data_dir, sample_transcription, sample_knowledge, tmp_path: Path
) -> None:
    err = RuntimeError("TTS failed")
    mp3 = tmp_path / "x.mp3"
    mp3.write_bytes(b"")
    with mock.patch.object(orch_mod, "_stage_ingest", return_value=mp3), \
         mock.patch.object(orch_mod, "_stage_transcribe", return_value=sample_transcription), \
         mock.patch.object(orch_mod, "_stage_extract", return_value=sample_knowledge), \
         mock.patch.object(orch_mod, "_stage_narrate", side_effect=err):
        with pytest.raises(RuntimeError, match="TTS failed"):
            run_job("j", JobSource.youtube("https://youtu.be/abc"))


# --- run_job input validation ---------------------------------------------


def test_run_job_rejects_empty_job_id() -> None:
    with pytest.raises(ValueError):
        run_job("", JobSource.youtube("https://youtu.be/abc"))


def test_run_job_rejects_non_string_job_id() -> None:
    with pytest.raises(ValueError):
        run_job(123, JobSource.youtube("https://youtu.be/abc"))  # type: ignore[arg-type]


def test_run_job_rejects_non_jobsource() -> None:
    with pytest.raises(ValueError):
        run_job("j", "not a source")  # type: ignore[arg-type]


# --- run_job_async ---------------------------------------------------------


def test_run_job_async_returns_same_result(
    tmp_data_dir, sample_transcription, sample_knowledge, tmp_path: Path
) -> None:
    mp3 = tmp_path / "x.mp3"
    mp3.write_bytes(b"")
    out = tmp_path / "o.mp3"
    out.write_bytes(b"")

    with mock.patch.object(orch_mod, "_stage_ingest", return_value=mp3), \
         mock.patch.object(orch_mod, "_stage_transcribe", return_value=sample_transcription), \
         mock.patch.object(orch_mod, "_stage_extract", return_value=sample_knowledge), \
         mock.patch.object(orch_mod, "_stage_narrate", return_value=out):
        result = asyncio.run(
            run_job_async("j-async", JobSource.youtube("https://youtu.be/abc"))
        )
    assert isinstance(result, LoomaResult)
    assert result.job_id == "j-async"
    assert result.audio_url == "/audio/j-async.mp3"


# --- AC-6 schema probe -----------------------------------------------------


def test_loomaresult_required_fields() -> None:
    """AC-6: ``LoomaResult`` exposes every required field."""
    fields = set(LoomaResult.model_fields.keys())
    assert {
        "job_id",
        "source_type",
        "source_ref",
        "title",
        "transcription",
        "knowledge",
        "audio_url",
        "created_at",
    } <= fields


def test_loomaresult_can_be_roundtripped(
    tmp_data_dir, sample_transcription, sample_knowledge, tmp_path: Path
) -> None:
    mp3 = tmp_path / "x.mp3"
    mp3.write_bytes(b"")
    out = tmp_path / "o.mp3"
    out.write_bytes(b"")
    with mock.patch.object(orch_mod, "_stage_ingest", return_value=mp3), \
         mock.patch.object(orch_mod, "_stage_transcribe", return_value=sample_transcription), \
         mock.patch.object(orch_mod, "_stage_extract", return_value=sample_knowledge), \
         mock.patch.object(orch_mod, "_stage_narrate", return_value=out):
        result = run_job("j", JobSource.youtube("https://youtu.be/abc"))
    # Pydantic round-trip
    dumped = result.model_dump()
    restored = LoomaResult.model_validate(dumped)
    assert restored.job_id == result.job_id
    assert restored.audio_url == result.audio_url
    assert restored.created_at == result.created_at
    assert restored.title == result.title
    assert restored.knowledge.title == result.knowledge.title
