"""Unit tests for ``app.pipeline.extract`` (AC-5).

All tests are pure unit tests — no live LLM calls. The
``_invoke_provider`` dispatch is patched in tests that exercise
:func:`extract_knowledge`, and the prompt loaders are not invoked
because we use the public function with a recorded fixture JSON.

The contract being verified:

* the LLM's response is parsed into a strict :class:`KnowledgeExtract`
  with all required fields,
* AC-5 length / count constraints (title <= 120, summary 3-5
  sentences, insights 5-10, narrative 150-400 words, filler_removed
  >= 0) are checked and surface as retryable errors when violated,
* chapter coverage is forced to ``[0, transcription.duration_seconds]``
  via :func:`_snap_chapters`,
* a single retry happens on schema failure, and a second failure
  raises :class:`LLMSchemaError`,
* provider selection honors arg > env > default and falls back across
  providers when keys are missing,
* missing API keys raise :class:`ValueError` (the startup guard
  AC-14 surfaces this to the user as a clear, actionable error).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from app.models import (
    Chapter,
    KnowledgeExtract,
    TranscriptSegment,
    TranscriptionResult,
)
from app.pipeline import extract as extract_mod
from app.pipeline.extract import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TEMPERATURE,
    LLMSchemaError,
    _call_anthropic,
    _call_openai,
    _count_sentences,
    _force_contiguous,
    _invoke_provider,
    _load_prompt,
    _narrative_word_count,
    _parse_json,
    _post_validate,
    _resolve_provider,
    _render_user_prompt,
    _segment_anchors,
    _snap_chapters,
    _truncate_transcript,
    extract_knowledge,
)


# --- Fixtures ---------------------------------------------------------------


FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def sample_transcription() -> TranscriptionResult:
    """A short transcription whose duration is 600 seconds."""
    raw = json.loads(
        (FIXTURES / "sample_transcription.json").read_text(encoding="utf-8")
    )
    return TranscriptionResult(
        transcript=raw["transcript"],
        segments=[TranscriptSegment(**s) for s in raw["segments"]],
        language=raw["language"],
        duration_seconds=raw["duration_seconds"],
    )


@pytest.fixture
def sample_llm_response() -> dict:
    """A recorded (offline) LLM response fixture."""
    return json.loads(
        (FIXTURES / "llm_extract_response.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def valid_extract(sample_llm_response) -> KnowledgeExtract:
    """A pre-validated :class:`KnowledgeExtract` from the fixture."""
    return KnowledgeExtract.model_validate(sample_llm_response)


# --- Prompt loading + rendering --------------------------------------------


def test_system_prompt_file_exists() -> None:
    """The system prompt is a real file on disk."""
    p = extract_mod._SYSTEM_PROMPT_PATH
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    # Must mention each of the required top-level fields
    for field in ("title", "summary", "insights", "chapters", "narrative", "filler_removed"):
        assert field in text


def test_user_prompt_file_exists() -> None:
    p = extract_mod._USER_PROMPT_PATH
    assert p.exists()


def test_load_prompt_reads_file(tmp_path: Path) -> None:
    f = tmp_path / "p.txt"
    f.write_text("hello")
    assert _load_prompt(f) == "hello"


def test_segment_anchors_renders_numbered_lines(sample_transcription: TranscriptionResult) -> None:
    rendered = _segment_anchors(sample_transcription)
    assert "[000]" in rendered
    assert "[006]" in rendered
    assert "0.0-30.0" in rendered


def test_segment_anchors_handles_empty_segments() -> None:
    t = TranscriptionResult(transcript="", segments=[], language="en", duration_seconds=0.0)
    assert _segment_anchors(t) == "(no segments available)"


def test_truncate_transcript_under_limit() -> None:
    text, was_truncated = _truncate_transcript("short text")
    assert text == "short text"
    assert was_truncated is False


def test_truncate_transcript_over_limit() -> None:
    text, was_truncated = _truncate_transcript("x" * (extract_mod._MAX_TRANSCRIPT_CHARS + 10))
    assert was_truncated is True
    assert text.startswith("x" * 100)
    assert "truncated" in text


def test_render_user_prompt_includes_metadata(sample_transcription: TranscriptionResult) -> None:
    rendered = _render_user_prompt(sample_transcription)
    assert "language:        en" in rendered
    assert "duration:        600.0 seconds" in rendered
    assert "transcript_chars:" in rendered
    assert "[000]" in rendered  # segment anchors present
    assert "Um, today" in rendered  # transcript body present


def test_render_user_prompt_can_append_retry_notice(
    sample_transcription: TranscriptionResult,
) -> None:
    rendered = _render_user_prompt(sample_transcription, retry_notice="(RETRY)")
    assert "(RETRY)" in rendered


# --- JSON parsing -----------------------------------------------------------


def test_parse_json_strips_markdown_fences() -> None:
    raw = "```json\n{\"title\": \"x\"}\n```"
    parsed = _parse_json(raw)
    assert parsed == {"title": "x"}


def test_parse_json_strips_fences_with_no_lang_tag() -> None:
    raw = "```\n{\"title\": \"x\"}\n```"
    parsed = _parse_json(raw)
    assert parsed == {"title": "x"}


def test_parse_json_passes_through_bare_json() -> None:
    parsed = _parse_json('{"title": "x"}')
    assert parsed == {"title": "x"}


def test_parse_json_raises_on_invalid() -> None:
    with pytest.raises(json.JSONDecodeError):
        _parse_json("not json at all")


# --- Validation helpers -----------------------------------------------------


def test_count_sentences_simple() -> None:
    assert _count_sentences("One. Two. Three.") == 3
    assert _count_sentences("Just one.") == 1


def test_count_sentences_strips_abbreviations() -> None:
    # e.g., i.e., etc. should not be counted as sentence ends
    assert _count_sentences("We use e.g. this. And i.e. that.") == 2


def test_narrative_word_count_basic() -> None:
    assert _narrative_word_count("a b c d e") == 5


def test_post_validate_passes_for_valid_extract(
    valid_extract: KnowledgeExtract, sample_transcription: TranscriptionResult
) -> None:
    assert _post_validate(valid_extract, sample_transcription.duration_seconds) == []


def test_post_validate_flags_long_title(
    valid_extract: KnowledgeExtract, sample_transcription: TranscriptionResult
) -> None:
    # Use model_construct to bypass Pydantic's max_length so we can
    # exercise the post-validator. The rest of the fields are kept valid.
    dump = valid_extract.model_dump()
    dump["title"] = "x" * 121
    bad = KnowledgeExtract.model_construct(**dump)
    issues = _post_validate(bad, sample_transcription.duration_seconds)
    assert any("title" in i for i in issues)


def test_post_validate_flags_short_summary(
    valid_extract: KnowledgeExtract, sample_transcription: TranscriptionResult
) -> None:
    dump = valid_extract.model_dump()
    dump["summary"] = "One."
    bad = KnowledgeExtract.model_construct(**dump)
    issues = _post_validate(bad, sample_transcription.duration_seconds)
    assert any("summary" in i for i in issues)


def test_post_validate_flags_too_few_insights(
    valid_extract: KnowledgeExtract, sample_transcription: TranscriptionResult
) -> None:
    dump = valid_extract.model_dump()
    dump["insights"] = ["Only one insight."]
    bad = KnowledgeExtract.model_construct(**dump)
    issues = _post_validate(bad, sample_transcription.duration_seconds)
    assert any("insights" in i for i in issues)


def test_post_validate_flags_too_many_insights(
    valid_extract: KnowledgeExtract, sample_transcription: TranscriptionResult
) -> None:
    dump = valid_extract.model_dump()
    dump["insights"] = [f"Insight {i}." for i in range(11)]
    bad = KnowledgeExtract.model_construct(**dump)
    issues = _post_validate(bad, sample_transcription.duration_seconds)
    assert any("insights" in i for i in issues)


def test_post_validate_flags_short_narrative(
    valid_extract: KnowledgeExtract, sample_transcription: TranscriptionResult
) -> None:
    dump = valid_extract.model_dump()
    dump["narrative"] = "Too short."
    bad = KnowledgeExtract.model_construct(**dump)
    issues = _post_validate(bad, sample_transcription.duration_seconds)
    assert any("narrative" in i for i in issues)


def test_post_validate_flags_long_narrative(
    valid_extract: KnowledgeExtract, sample_transcription: TranscriptionResult
) -> None:
    dump = valid_extract.model_dump()
    dump["narrative"] = " ".join(["word"] * 501)
    bad = KnowledgeExtract.model_construct(**dump)
    issues = _post_validate(bad, sample_transcription.duration_seconds)
    assert any("narrative" in i for i in issues)


# --- Chapter snapping -------------------------------------------------------


def test_snap_chapters_first_starts_at_zero() -> None:
    chapters = [
        Chapter(start_seconds=2.0,  end_seconds=10.0, title="a"),
        Chapter(start_seconds=10.0, end_seconds=20.0, title="b"),
    ]
    out = _snap_chapters(chapters, segments=[], duration=20.0)
    assert out[0].start_seconds == 0.0
    assert out[-1].end_seconds == 20.0


def test_snap_chapters_last_ends_at_duration() -> None:
    chapters = [
        Chapter(start_seconds=0.0, end_seconds=8.0, title="a"),
        Chapter(start_seconds=8.0, end_seconds=18.0, title="b"),
    ]
    out = _snap_chapters(chapters, segments=[], duration=20.0)
    assert out[-1].end_seconds == 20.0


def test_snap_chapters_snaps_to_segments() -> None:
    segments = [
        TranscriptSegment(start=0.0, end=30.0, text="a"),
        TranscriptSegment(start=30.0, end=60.0, text="b"),
    ]
    chapters = [
        Chapter(start_seconds=2.0, end_seconds=33.0, title="a"),
        Chapter(start_seconds=33.0, end_seconds=58.0, title="b"),
    ]
    out = _snap_chapters(chapters, segments=segments, duration=60.0)
    # First chapter: start forced to 0.0, end snapped to nearest of
    # {0, 30, 60, durations} -> 33.0 -> 30.0.
    assert out[0].start_seconds == 0.0
    assert out[0].end_seconds == 30.0
    # Last chapter: end forced to 60.0, start snapped from 33.0 -> 30.0.
    assert out[-1].start_seconds == 30.0
    assert out[-1].end_seconds == 60.0


def test_snap_chapters_covers_full_range() -> None:
    segments = [
        TranscriptSegment(start=0.0, end=20.0, text="a"),
        TranscriptSegment(start=20.0, end=40.0, text="b"),
        TranscriptSegment(start=40.0, end=60.0, text="c"),
    ]
    chapters = [
        Chapter(start_seconds=5.0,  end_seconds=25.0, title="a"),
        Chapter(start_seconds=25.0, end_seconds=45.0, title="b"),
        Chapter(start_seconds=45.0, end_seconds=58.0, title="c"),
    ]
    out = _snap_chapters(chapters, segments=segments, duration=60.0)
    # After snapping: first chapter 0-20, middle 20-40, last 40-60.
    assert out[0].start_seconds == 0.0
    assert out[-1].end_seconds == 60.0
    # No gaps, no overlaps.
    for i in range(len(out) - 1):
        assert out[i].end_seconds == out[i + 1].start_seconds


def test_snap_chapters_handles_empty() -> None:
    assert _snap_chapters([], segments=[], duration=10.0) == []


def test_force_contiguous_tiles_range() -> None:
    chapters = [
        Chapter(start_seconds=0.0,  end_seconds=15.0, title="a"),
        Chapter(start_seconds=10.0, end_seconds=25.0, title="b"),  # overlap
        Chapter(start_seconds=20.0, end_seconds=40.0, title="c"),  # gap
    ]
    out = _force_contiguous(chapters, duration=40.0)
    assert out[0].start_seconds == 0.0
    assert out[-1].end_seconds == 40.0
    for i in range(len(out) - 1):
        assert out[i].end_seconds == out[i + 1].start_seconds


# --- Provider resolution ---------------------------------------------------


def test_resolve_provider_prefers_explicit_arg() -> None:
    assert _resolve_provider("openai") == "openai"


def test_resolve_provider_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert _resolve_provider() == "openai"


def test_resolve_provider_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert _resolve_provider() == DEFAULT_LLM_PROVIDER == "anthropic"


def test_resolve_provider_unknown_value_warns(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert _resolve_provider() == "anthropic"


def test_resolve_provider_falls_back_when_key_missing(monkeypatch) -> None:
    """If anthropic is requested but only OpenAI key is set, use OpenAI."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert _resolve_provider() == "openai"


def test_resolve_provider_openai_fallback(monkeypatch) -> None:
    """If openai is requested but only Anthropic key is set, use Anthropic."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _resolve_provider() == "anthropic"


def test_resolve_provider_no_keys_returns_requested(monkeypatch) -> None:
    """When no key for either is set, return the requested provider so
    the caller can raise a clear error."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _resolve_provider() == "anthropic"


# --- extract_knowledge happy path -----------------------------------------


def test_extract_knowledge_happy_path_anthropic(
    sample_transcription: TranscriptionResult, sample_llm_response: dict
) -> None:
    with mock.patch.object(
        extract_mod,
        "_invoke_provider",
        return_value=json.dumps(sample_llm_response),
    ) as invoke:
        with mock.patch.dict(
            os.environ, {"ANTHROPIC_API_KEY": "test"}, clear=False
        ):
            result = extract_knowledge(sample_transcription)

    invoke.assert_called_once()
    assert isinstance(result, KnowledgeExtract)
    assert result.title == sample_llm_response["title"]
    assert len(result.insights) >= 5
    assert len(result.chapters) >= 1
    assert result.chapters[0].start_seconds == 0.0
    assert result.chapters[-1].end_seconds == sample_transcription.duration_seconds


def test_extract_knowledge_uses_openai_when_requested(
    sample_transcription: TranscriptionResult, sample_llm_response: dict
) -> None:
    with mock.patch.object(
        extract_mod, "_invoke_provider", return_value=json.dumps(sample_llm_response)
    ) as invoke:
        with mock.patch.dict(
            os.environ, {"OPENAI_API_KEY": "test"}, clear=False
        ):
            result = extract_knowledge(sample_transcription, provider="openai")
    invoke.assert_called_once()
    # Provider arg should have been honored.
    assert invoke.call_args.args[0] == "openai"
    assert isinstance(result, KnowledgeExtract)


def test_extract_knowledge_strips_markdown_fences(
    sample_transcription: TranscriptionResult, sample_llm_response: dict
) -> None:
    fenced = "```json\n" + json.dumps(sample_llm_response) + "\n```"
    with mock.patch.object(extract_mod, "_invoke_provider", return_value=fenced):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}, clear=False):
            result = extract_knowledge(sample_transcription)
    assert result.title == sample_llm_response["title"]


def test_extract_knowledge_no_keys_raises_value_error(
    sample_transcription: TranscriptionResult, monkeypatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError) as excinfo:
        extract_knowledge(sample_transcription)
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)
    assert "OPENAI_API_KEY" in str(excinfo.value)


# --- extract_knowledge retry / failure --------------------------------------


def test_extract_knowledge_retries_once_on_schema_failure(
    sample_transcription: TranscriptionResult,
) -> None:
    """First response is bad, second is good -> succeeds after one retry."""
    bad = json.dumps({"title": "ok", "summary": "x"})  # missing fields
    good = json.dumps(json.loads(
        (FIXTURES / "llm_extract_response.json").read_text(encoding="utf-8")
    ))
    with mock.patch.object(
        extract_mod, "_invoke_provider", side_effect=[bad, good]
    ) as invoke:
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}, clear=False):
            result = extract_knowledge(sample_transcription)
    # Two attempts, then success.
    assert invoke.call_count == 2
    assert isinstance(result, KnowledgeExtract)
    # Second call should include the retry notice.
    second_prompt = invoke.call_args_list[1].args[2]
    assert "RETRY NOTICE" in second_prompt


def test_extract_knowledge_raises_after_two_failures(
    sample_transcription: TranscriptionResult,
) -> None:
    """Both attempts return bad JSON -> LLMSchemaError."""
    bad = "not json at all"
    with mock.patch.object(extract_mod, "_invoke_provider", return_value=bad):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}, clear=False):
            with pytest.raises(LLMSchemaError) as excinfo:
                extract_knowledge(sample_transcription)
    assert "2 attempts" in str(excinfo.value)


def test_extract_knowledge_raises_after_two_post_validation_failures(
    sample_transcription: TranscriptionResult,
) -> None:
    """Both attempts return valid JSON but violate AC-5 length constraints."""
    # Title too long + summary too short -> post-validation failure.
    bad_payload = {
        "title": "x" * 200,
        "summary": "One.",
        "insights": ["a", "b", "c", "d", "e"],
        "chapters": [
            {"start_seconds": 0.0, "end_seconds": 600.0, "title": "all"},
        ],
        "narrative": "too short",
        "filler_removed": 0,
    }
    with mock.patch.object(
        extract_mod, "_invoke_provider", return_value=json.dumps(bad_payload)
    ) as invoke:
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}, clear=False):
            with pytest.raises(LLMSchemaError):
                extract_knowledge(sample_transcription)
    assert invoke.call_count == 2


def test_extract_knowledge_raises_on_unknown_provider(
    sample_transcription: TranscriptionResult, monkeypatch
) -> None:
    """Defensive: if _resolve_provider somehow returns garbage, raise."""
    monkeypatch.setattr(extract_mod, "_resolve_provider", lambda *_a, **_k: "garbage")
    with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}, clear=False):
        with pytest.raises(LLMSchemaError) as excinfo:
            extract_knowledge(sample_transcription)
    assert "Unknown LLM provider" in str(excinfo.value)


# --- _invoke_provider dispatch --------------------------------------------


def test_invoke_provider_anthropic(sample_llm_response: dict) -> None:
    payload = json.dumps(sample_llm_response)
    with mock.patch.object(extract_mod, "_call_anthropic", return_value=payload) as call:
        out = _invoke_provider("anthropic", "system", "user")
    call.assert_called_once_with("system", "user")
    assert out == payload


def test_invoke_provider_openai(sample_llm_response: dict) -> None:
    payload = json.dumps(sample_llm_response)
    with mock.patch.object(extract_mod, "_call_openai", return_value=payload) as call:
        out = _invoke_provider("openai", "system", "user")
    call.assert_called_once_with("system", "user")
    assert out == payload


def test_invoke_provider_unknown_raises() -> None:
    with pytest.raises(LLMSchemaError):
        _invoke_provider("garbage", "system", "user")


# --- _call_anthropic / _call_openai ---------------------------------------


def test_call_anthropic_passes_args_to_client(
    sample_llm_response: dict, monkeypatch
) -> None:
    """Smoke: the Anthropic call site is wired with the expected kwargs."""
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    fake_client = mock.MagicMock()
    block = mock.MagicMock()
    block.text = json.dumps(sample_llm_response)
    fake_client.messages.create.return_value.content = [block]
    with mock.patch.object(extract_mod, "_get_anthropic_client", return_value=fake_client):
        out = _call_anthropic("system", "user")
    create = fake_client.messages.create
    create.assert_called_once()
    kwargs = create.call_args.kwargs
    assert kwargs["system"] == "system"
    assert kwargs["messages"] == [{"role": "user", "content": "user"}]
    assert kwargs["max_tokens"] == DEFAULT_MAX_TOKENS
    assert kwargs["temperature"] == DEFAULT_TEMPERATURE
    assert kwargs["model"] == DEFAULT_ANTHROPIC_MODEL
    assert out == json.dumps(sample_llm_response)


def test_call_anthropic_uses_env_model(
    sample_llm_response: dict, monkeypatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    fake_client = mock.MagicMock()
    block = mock.MagicMock()
    block.text = json.dumps(sample_llm_response)
    fake_client.messages.create.return_value.content = [block]
    with mock.patch.dict(os.environ, {"ANTHROPIC_MODEL": "claude-3-haiku-20240307"}):
        with mock.patch.object(
            extract_mod, "_get_anthropic_client", return_value=fake_client
        ):
            _call_anthropic("s", "u")
    assert fake_client.messages.create.call_args.kwargs["model"] == "claude-3-haiku-20240307"


def test_call_openai_passes_args_to_client(
    sample_llm_response: dict, monkeypatch
) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    fake_client = mock.MagicMock()
    fake_client.chat.completions.create.return_value.choices = [
        mock.MagicMock(message=mock.MagicMock(content=json.dumps(sample_llm_response)))
    ]
    with mock.patch.object(extract_mod, "_get_openai_client", return_value=fake_client):
        out = _call_openai("system", "user")
    create = fake_client.chat.completions.create
    create.assert_called_once()
    kwargs = create.call_args.kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert kwargs["max_tokens"] == DEFAULT_MAX_TOKENS
    assert kwargs["temperature"] == DEFAULT_TEMPERATURE
    assert kwargs["model"] == DEFAULT_OPENAI_MODEL
    assert out == json.dumps(sample_llm_response)


def test_call_openai_handles_null_content(sample_llm_response: dict) -> None:
    fake_client = mock.MagicMock()
    fake_client.chat.completions.create.return_value.choices = [
        mock.MagicMock(message=mock.MagicMock(content=None))
    ]
    with mock.patch.object(extract_mod, "_get_openai_client", return_value=fake_client):
        out = _call_openai("s", "u")
    assert out == ""


# --- Models shape (AC-5) --------------------------------------------------


def test_knowledge_extract_required_fields() -> None:
    """AC-5 requires title, summary, insights, chapters, narrative, filler_removed."""
    fields = set(KnowledgeExtract.model_fields.keys())
    assert {"title", "summary", "insights", "chapters", "narrative", "filler_removed"} <= fields


def test_chapter_required_fields() -> None:
    fields = set(Chapter.model_fields.keys())
    assert {"start_seconds", "end_seconds", "title"} <= fields


def test_knowledge_extract_insights_length_5_to_10() -> None:
    """Pydantic enforces the 5-10 insights range."""
    base = json.loads(
        (FIXTURES / "llm_extract_response.json").read_text(encoding="utf-8")
    )
    too_few = {**base, "insights": base["insights"][:4]}
    with pytest.raises(Exception):
        KnowledgeExtract.model_validate(too_few)


def test_knowledge_extract_title_max_length_120() -> None:
    base = json.loads(
        (FIXTURES / "llm_extract_response.json").read_text(encoding="utf-8")
    )
    too_long = {**base, "title": base["title"] + "x" * 200}
    with pytest.raises(Exception):
        KnowledgeExtract.model_validate(too_long)


def test_knowledge_extract_filler_removed_non_negative() -> None:
    base = json.loads(
        (FIXTURES / "llm_extract_response.json").read_text(encoding="utf-8")
    )
    bad = {**base, "filler_removed": -1}
    with pytest.raises(Exception):
        KnowledgeExtract.model_validate(bad)


def test_error_code_constant() -> None:
    from app.models import CODE_LLM_SCHEMA_ERROR

    assert CODE_LLM_SCHEMA_ERROR == "LLM_SCHEMA_ERROR"
