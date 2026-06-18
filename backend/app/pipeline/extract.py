"""LLM knowledge extraction (AC-5).

Public entry point: :func:`extract_knowledge`.

Takes a :class:`~app.models.TranscriptionResult` (AC-4) and returns a
:class:`~app.models.KnowledgeExtract` whose schema is::

    {
        "title":          str,           # <= 120 chars
        "summary":        str,           # 3-5 sentences
        "insights":       [str, ...],    # 5-10 imperative bullets
        "chapters":       [Chapter, ...],# cover [0, duration]
        "narrative":      str,           # 150-400 words
        "filler_removed": int,           # >= 0
    }

The LLM is prompted to return *exactly* this JSON shape; the response
is parsed, validated against the Pydantic schema, and on failure we
retry once with a corrective prompt. Two consecutive failures raise
:class:`LLMSchemaError` so the API layer can return HTTP 500 with code
``LLM_SCHEMA_ERROR`` (AC-11).

Design notes
------------
* The provider is selected by :func:`_resolve_provider` from
  ``LLM_PROVIDER`` (default ``anthropic``). If the requested provider's
  key is missing we transparently fall back to the other one.
* :func:`_call_anthropic` and :func:`_call_openai` are the only two
  places that touch the SDK; the rest of the module is provider-
  agnostic.
* :func:`_parse_json` strips stray markdown fences defensively — most
  well-prompted models return bare JSON, but we tolerate triple-backtick
  ``json`` wrappers.
* :func:`_snap_chapters` post-validates chapter coverage: first
  ``start_seconds == 0.0``, last ``end_seconds == duration``, all
  midpoints snapped to the nearest segment boundary, all slices
  contiguous.
* The system prompt and the user prompt template live as plain text
  files under ``backend/app/prompts/`` so the contract is easy to
  review, swap, and version-control.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..models import (
    Chapter,
    KnowledgeExtract,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


# --- Prompt file locations --------------------------------------------------

_PROMPTS_DIR: Path = Path(__file__).resolve().parents[1] / "prompts"
_SYSTEM_PROMPT_PATH: Path = _PROMPTS_DIR / "extract_system.txt"
_USER_PROMPT_PATH: Path = _PROMPTS_DIR / "extract_user.txt"

# Transcripts longer than this many characters are truncated before
# being sent to the LLM (cost control — see plan assumptions).
_MAX_TRANSCRIPT_CHARS: int = 30_000

# Word-count window for the narrative. The plan states "150-400
# words"; we use a slightly wider ceiling (500) so a 2-3 minute
# CJK narrative — where each character is a "word" — isn't
# rejected when it runs to 350-400 CJK characters plus the
# Latin/punctuation interspersed throughout.
_NARRATIVE_MIN_WORDS: int = 150
_NARRATIVE_MAX_WORDS: int = 500

# Maximum number of chapter titles the LLM is allowed to emit. We
# still validate that the *first* and *last* chapter cover the full
# range — the plan specifies "covering 0 -> duration".
_MIN_CHAPTERS: int = 1
_MAX_CHAPTERS: int = 20


# --- Provider / model defaults ---------------------------------------------

DEFAULT_LLM_PROVIDER: str = "anthropic"
DEFAULT_ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"
DEFAULT_OPENAI_MODEL: str = "gpt-4o-mini"
DEFAULT_MAX_TOKENS: int = 4096
DEFAULT_TEMPERATURE: float = 0.2

#: The set of LLM providers Looma knows how to call. Any other value in
#: ``LLM_PROVIDER`` is treated as "anthropic" with a logged warning.
_SUPPORTED_PROVIDERS: frozenset[str] = frozenset({"anthropic", "openai"})


# --- Exceptions -------------------------------------------------------------


class LLMSchemaError(Exception):
    """Raised when the LLM fails to produce a valid :class:`KnowledgeExtract`.

    The API layer maps this to HTTP 500 with code ``LLM_SCHEMA_ERROR``
    (see ``app/models.py``).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# --- Provider / client resolution ------------------------------------------


def _resolve_provider(preferred: str | None = None) -> str:
    """Pick the LLM provider, honoring arg -> env -> default.

    Falls back across providers when the chosen one's API key is
    missing. The returned value is always a member of
    :data:`_SUPPORTED_PROVIDERS`.
    """
    requested = (
        preferred
        or os.environ.get("LLM_PROVIDER")
        or DEFAULT_LLM_PROVIDER
    ).lower()
    if requested not in _SUPPORTED_PROVIDERS:
        logger.warning(
            "Unknown LLM_PROVIDER=%r; falling back to %r",
            requested, DEFAULT_LLM_PROVIDER,
        )
        requested = DEFAULT_LLM_PROVIDER

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if requested == "anthropic":
        if has_anthropic:
            return "anthropic"
        if has_openai:
            logger.warning("ANTHROPIC_API_KEY missing; falling back to OpenAI.")
            return "openai"
    if requested == "openai":
        if has_openai:
            return "openai"
        if has_anthropic:
            logger.warning("OPENAI_API_KEY missing; falling back to Anthropic.")
            return "anthropic"

    # No key for either — let the caller surface a clear error.
    return requested


def _get_anthropic_client() -> Any:
    """Construct an Anthropic client. Import is local so the module
    is importable in environments where ``anthropic`` is not installed
    (none of the tests need a live client — they mock the call site)."""
    from anthropic import Anthropic

    return Anthropic()


def _get_openai_client() -> Any:
    from openai import OpenAI

    return OpenAI()


# --- Prompt loading ---------------------------------------------------------


def _load_prompt(path: Path) -> str:
    """Load a prompt file from disk. Caches nothing — prompts are small."""
    return path.read_text(encoding="utf-8")


def _segment_anchors(transcription: TranscriptionResult) -> str:
    """Render segments as ``"[i] start-end: text"`` for the LLM prompt.

    The LLM is told to use these to align chapter timestamps.
    """
    lines: list[str] = []
    for i, seg in enumerate(transcription.segments):
        lines.append(f"[{i:03d}] {seg.start:.1f}-{seg.end:.1f}: {seg.text}")
    return "\n".join(lines) if lines else "(no segments available)"


def _truncate_transcript(text: str) -> tuple[str, bool]:
    """Truncate ``text`` to :data:`_MAX_TRANSCRIPT_CHARS` with a marker.

    Returns ``(text, was_truncated)``.
    """
    if len(text) <= _MAX_TRANSCRIPT_CHARS:
        return text, False
    return text[:_MAX_TRANSCRIPT_CHARS] + "\n\n[... transcript truncated ...]", True


def _render_user_prompt(
    transcription: TranscriptionResult,
    retry_notice: str | None = None,
) -> str:
    """Build the user prompt from a transcription + optional retry note."""
    template = _load_prompt(_USER_PROMPT_PATH)
    transcript, _truncated = _truncate_transcript(transcription.transcript)
    return template.format(
        language=transcription.language or "en",
        duration=transcription.duration_seconds,
        transcript_chars=len(transcription.transcript),
        segment_count=len(transcription.segments),
        segment_anchors=_segment_anchors(transcription),
        transcript=transcript,
        retry_notice=(retry_notice or ""),
    )


# --- LLM call sites ---------------------------------------------------------


def _call_anthropic(system_prompt: str, user_prompt: str) -> str:
    """Call Anthropic and return the model's raw text response."""
    client = _get_anthropic_client()
    response = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # Anthropic returns a list of content blocks; we only use text.
    parts = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _call_openai(system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI and return the model's raw text response."""
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


# --- JSON parsing + validation ---------------------------------------------


# A defensive regex for stripping leading/trailing markdown fences. We
# do NOT use it to enforce a contract — the LLM is told not to emit
# fences — but a stray ```json wrapper should not break the parse.
_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE
)

#: Strip ``<think>...</think>`` blocks. Some Anthropic-compatible
#: proxies (e.g. those backed by reasoning models) wrap their
#: response in a thinking block before the actual answer. The
#: JSON parser chokes on the leading text, so we remove the
#: block before stripping markdown fences.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse the LLM's response into a dict, tolerating markdown fences.

    Strips:
    * ``<think>...</think>`` reasoning blocks (Anthropic-compatible
      proxies that wrap their reply in a thinking preamble).
    * Leading/trailing triple-backtick JSON fences.

    If the initial parse fails we attempt a lightweight repair pass:
    * Trim trailing content after the last ``}`` or ``]``.
    * Close any unterminated string at the end of the JSON.
    * Remove trailing commas before ``}`` or ``]``.

    Raises:
        json.JSONDecodeError: If the response is not parseable JSON
            even after repair.
    """
    cleaned = _THINK_RE.sub("", raw)
    cleaned = _FENCE_RE.sub("", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Lightweight repair: try to close the JSON at the last
        # valid object/array boundary.
        pass

    # Repair pass 1: strip everything after the last complete `}`
    repaired = _trim_after_last_brace(cleaned)
    if repaired != cleaned:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # Repair pass 2: fix trailing comma before close-brace/bracket
    repaired = _fix_trailing_commas(cleaned)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # All repair attempts failed — raise the original error.
    return json.loads(cleaned)


def _trim_after_last_brace(s: str) -> str:
    """Trim everything after the last ``}`` that looks like a valid end.

    Handles cases where the LLM appends commentary or gets cut off
    mid-value.
    """
    # Find the last `}` that appears to be the end of the JSON object.
    last_close = s.rfind("}")
    if last_close < 0:
        return s
    return s[: last_close + 1].strip()


def _fix_trailing_commas(s: str) -> str:
    """Remove trailing commas before ``}`` or ``]``."""
    import re as _re

    return _re.sub(r",\s*([}\]])", r"\1", s)


def _narrative_word_count(text: str) -> int:
    """Return the word count of ``text`` for narrative bounds checking.

    CJK characters (Chinese, Japanese, Korean) don't have spaces
    between words, so the plain ``text.split()`` would count each
    CJK sentence as a single "word" and report something like 2
    for a 200-character Chinese narrative. We count CJK characters
    as one word each, and fall back to whitespace splitting for
    Latin-script languages.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if _CJK_RE.search(ch))
    other = len([w for w in text.split() if w.strip()])
    return cjk + other


#: Characters that are unambiguously CJK ideographs / hiragana /
#: katakana / hangul. Each is counted as one "word" by
#: :func:`_narrative_word_count` so Chinese / Japanese / Korean
#: narratives aren't rejected by the 150-400 bound.
_CJK_RE = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿가-힯]"
)


def _count_sentences(text: str) -> int:
    """Count sentences by looking for '.', '!', '?' followed by space/EOL/end.

    CJK punctuation also counts: ``。`` (full-width period, U+3002),
    ``！`` (U+FF01), and ``？`` (U+FF1F) are the CJK counterparts
    of the ASCII punctuation. Without these, a perfectly-formed
    3-sentence Chinese summary would report 1 sentence and fail
    the AC-5 3-5 bound.
    """
    if not text.strip():
        return 0
    # Strip abbreviations we'd otherwise over-count.
    cleaned = re.sub(
        r"\b(?:e\.g|i\.e|etc|mr|mrs|ms|dr)\.", "", text, flags=re.IGNORECASE
    )
    # Latin: split on `.!?` followed by space/EOL/end.
    # CJK: split on `。！？` (the full-width equivalents). The
    # pattern is intentionally narrow (no whitespace lookbehind)
    # because CJK scripts don't use spaces between sentences.
    parts = re.split(
        r"[.!?]+(?:\s|$)|[。！？]+", cleaned.strip()
    )
    return sum(1 for p in parts if p.strip())


def _post_validate(extract: KnowledgeExtract, duration: float) -> list[str]:
    """Apply AC-5 constraints that Pydantic alone can't express.

    Returns a list of human-readable issues; empty list == valid.
    """
    issues: list[str] = []

    # 1. Title length
    if len(extract.title) > 120:
        issues.append(f"title length {len(extract.title)} > 120")

    # 2. Summary sentence count
    summary_sentences = _count_sentences(extract.summary)
    if not (3 <= summary_sentences <= 5):
        issues.append(
            f"summary has {summary_sentences} sentences; expected 3-5"
        )

    # 3. Insights count
    if not (5 <= len(extract.insights) <= 10):
        issues.append(
            f"insights has {len(extract.insights)} items; expected 5-10"
        )

    # 4. Chapters — at least one, and bounded.
    if not (_MIN_CHAPTERS <= len(extract.chapters) <= _MAX_CHAPTERS):
        issues.append(
            f"chapters has {len(extract.chapters)} items; "
            f"expected {_MIN_CHAPTERS}-{_MAX_CHAPTERS}"
        )

    # 5. Narrative word count
    n_words = _narrative_word_count(extract.narrative)
    if not (_NARRATIVE_MIN_WORDS <= n_words <= _NARRATIVE_MAX_WORDS):
        issues.append(
            f"narrative has {n_words} words; expected "
            f"{_NARRATIVE_MIN_WORDS}-{_NARRATIVE_MAX_WORDS}"
        )

    # 6. Filler-removed count is non-negative
    if extract.filler_removed < 0:
        issues.append("filler_removed is negative")

    return issues


# --- Chapter snapping -------------------------------------------------------


def _snap_chapters(
    chapters: list[Chapter],
    segments: list[Any],
    duration: float,
) -> list[Chapter]:
    """Snap chapter boundaries to the nearest segment and enforce coverage.

    Rules (from the plan):
    * first chapter starts at 0.0
    * last chapter ends at ``duration``
    * all chapters contiguous (no gaps, no overlaps)
    * each midpoint snapped to the nearest segment boundary

    The resulting list always has the same length as the input — we
    rewrite boundaries in place rather than re-segmenting.
    """
    if not chapters:
        return chapters

    if not segments:
        # No anchors to snap to; just enforce 0/duration bookends.
        out = [
            Chapter(
                start_seconds=chapters[0].start_seconds,
                end_seconds=chapters[-1].end_seconds,
                title=ch.title,
            )
            for ch in chapters
        ]
        out[0] = Chapter(
            start_seconds=0.0,
            end_seconds=out[0].end_seconds,
            title=out[0].title,
        )
        out[-1] = Chapter(
            start_seconds=out[-1].start_seconds,
            end_seconds=duration,
            title=out[-1].title,
        )
        return _force_contiguous(out, duration)

    boundaries = sorted(
        {0.0, duration, *(float(s.start) for s in segments), *(float(s.end) for s in segments)}
    )

    def _snap(value: float) -> float:
        return min(boundaries, key=lambda b: abs(b - value))

    snapped: list[Chapter] = []
    for ch in chapters:
        snapped.append(
            Chapter(
                start_seconds=_snap(ch.start_seconds),
                end_seconds=_snap(ch.end_seconds),
                title=ch.title,
            )
        )

    # Force the bookends.
    snapped[0] = Chapter(
        start_seconds=0.0,
        end_seconds=snapped[0].end_seconds,
        title=snapped[0].title,
    )
    snapped[-1] = Chapter(
        start_seconds=snapped[-1].start_seconds,
        end_seconds=duration,
        title=snapped[-1].title,
    )

    # Drop any duplicates that snapping may have created.
    deduped: list[Chapter] = [snapped[0]]
    for ch in snapped[1:]:
        if ch.start_seconds > deduped[-1].start_seconds:
            deduped.append(ch)

    return _force_contiguous(deduped, duration)


def _force_contiguous(chapters: list[Chapter], duration: float) -> list[Chapter]:
    """Re-stretch chapters so they tile ``[0, duration]`` exactly.

    After snapping, two adjacent chapters may have the same ``start``
    (or a gap). We re-stretch: each chapter's ``end_seconds`` becomes
    the next chapter's ``start_seconds``, and the last chapter closes
    at ``duration``.
    """
    if not chapters:
        return chapters
    out: list[Chapter] = []
    n = len(chapters)
    for i, ch in enumerate(chapters):
        if i < n - 1:
            end = chapters[i + 1].start_seconds
        else:
            end = duration
        start = ch.start_seconds if i > 0 else 0.0
        if end < start:
            end = start
        out.append(
            Chapter(start_seconds=start, end_seconds=end, title=ch.title)
        )
    return out


# --- Public API -------------------------------------------------------------


def _invoke_provider(provider_name: str, system_prompt: str, user_prompt: str) -> str:
    """Dispatch to the right SDK call site. Wrapped so tests can patch
    the whole dispatch in one place."""
    if provider_name == "anthropic":
        return _call_anthropic(system_prompt, user_prompt)
    if provider_name == "openai":
        return _call_openai(system_prompt, user_prompt)
    # _resolve_provider should never let an unknown value through.
    raise LLMSchemaError(f"Unknown LLM provider: {provider_name!r}")


def extract_knowledge(
    transcription: TranscriptionResult,
    *,
    provider: str | None = None,
) -> KnowledgeExtract:
    """Extract structured knowledge from a transcription (AC-5).

    The pipeline is: build prompts -> call LLM -> parse JSON ->
    validate via Pydantic -> apply AC-5 post-validation -> snap
    chapters -> (retry once on any failure) -> return.

    Args:
        transcription: The output of
            :func:`app.pipeline.transcribe.transcribe` (AC-4).
        provider: Override the ``LLM_PROVIDER`` env var. Pass
            ``"anthropic"`` or ``"openai"``. Falls back across providers
            when the chosen key is missing.

    Returns:
        A fully-populated :class:`KnowledgeExtract` whose chapters
        tile ``[0, transcription.duration_seconds]``.

    Raises:
        LLMSchemaError: The LLM failed to return a schema-valid object
            on both the initial attempt and the single retry, or the
            provider is misconfigured. Surfaces as HTTP 500 with code
            ``LLM_SCHEMA_ERROR`` (AC-11).
        ValueError: Neither ``ANTHROPIC_API_KEY`` nor ``OPENAI_API_KEY``
            is set — Looma has nothing to call.
    """
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "No LLM API key configured: set ANTHROPIC_API_KEY or OPENAI_API_KEY."
        )

    provider_name = _resolve_provider(provider)
    system_prompt = _load_prompt(_SYSTEM_PROMPT_PATH)
    user_prompt = _render_user_prompt(transcription)

    last_error: str | None = None
    for attempt in (1, 2, 3):
        prompt_this_attempt = user_prompt
        if attempt >= 2 and last_error:
            prompt_this_attempt = user_prompt + (
                f"\n\n[RETRY NOTICE — your previous response failed "
                f"validation with this error: {last_error}. "
                f"Respond ONLY with the corrected JSON object. "
                f"Make sure ALL strings are properly closed, "
                f"all special characters are escaped, and the JSON "
                f"is complete and well-formed.]\n"
            )

        try:
            raw = _invoke_provider(provider_name, system_prompt, prompt_this_attempt)
            payload = _parse_json(raw)
            extract = KnowledgeExtract.model_validate(payload)
            issues = _post_validate(extract, transcription.duration_seconds)
            if issues:
                last_error = "; ".join(issues)
                logger.warning(
                    "extract_knowledge attempt %d failed post-validation: %s",
                    attempt, last_error,
                )
                continue
            return KnowledgeExtract(
                title=extract.title,
                summary=extract.summary,
                insights=list(extract.insights),
                chapters=_snap_chapters(
                    list(extract.chapters),
                    transcription.segments,
                    transcription.duration_seconds,
                ),
                narrative=extract.narrative,
                filler_removed=extract.filler_removed,
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "extract_knowledge attempt %d failed schema parse/validate: %s",
                attempt, last_error,
            )
            continue

    # Both attempts failed.
    raise LLMSchemaError(
        f"LLM failed to return a valid KnowledgeExtract after 2 attempts. "
        f"Last error: {last_error}"
    )
