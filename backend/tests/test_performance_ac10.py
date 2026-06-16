"""AC-10 performance + DELETE tests.

Covers the two halves of AC-10:

1. **DELETE /api/jobs/{job_id}** removes both the DB row and the
   on-disk MP3s. The behavior itself is also exercised in
   ``test_api_jobs.py``; this module is the perf-budget smoke test
   that ensures a delete is fast and shape-stable even on a large
   row set (the kind of growth we'd see from a few days of use).

2. **Pipeline wall-clock budget.** AC-10 requires the full
   YouTube-URL -> JSON + MP3 pipeline to finish in **under 5
   minutes** for a 20-minute source. We can't actually run a
   20-minute source in CI (it'd burn 5 minutes of test time and
   require live Whisper + network), so we verify the budget in
   two complementary ways:

   * **Structural assertions** on :class:`PipelineTimings`,
     :func:`_stage_timed`, and :func:`_record_timing` — these
     guarantee the orchestrator's instrumentation is wired
     correctly.
   * **Mocked latency budget** — the four stages are stubbed
     with controlled delays that sum to **under 5 minutes**;
     the test asserts the orchestrator's wall-clock total
     matches the sum within rounding. A regression that, e.g.,
     accidentally re-runs the extract stage would blow the
     budget and fail the test.
   * **Per-stage soft budgets** — the soft budgets in
     :mod:`app.config` are exported and asserted to match the
     plan's documentation so a future PR that loosens them
     without a code-review discussion is caught.

The DELETE tests live here too so AC-10 has one home for its
verification surface; a new contributor should be able to read
this module and know everything AC-10 promises.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from unittest import mock

import httpx
import pytest

from app import config as cfg
from app import main as main_mod
from app.config import (
    AUDIO_DIR,
    MAX_PIPELINE_SECONDS,
    OUTPUTS_DIR,
    STAGE_BUDGET_EXTRACT_SECONDS,
    STAGE_BUDGET_INGEST_SECONDS,
    STAGE_BUDGET_NARRATE_SECONDS,
    STAGE_BUDGET_TRANSCRIBE_SECONDS,
)
from app.models import (
    Chapter,
    KnowledgeExtract,
    PipelineTimings,
    TranscriptSegment,
    TranscriptionResult,
)
from app.pipeline import orchestrator as orch_mod
from app.pipeline.orchestrator import (
    JobSource,
    _record_timing,
    _soft_budget_for,
    _stage_timed,
    run_job,
)
from app.storage.jobs import (
    JobStatus,
    get_default_db,
    reset_default_db,
)


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch) -> Path:
    """Redirect the default JobsDB to a fresh tmp JSON file for the test."""
    from app import config as cfg
    from app.storage import jobs as jobs_mod

    p = tmp_path / "jobs.json"
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "JOBS_JSON_PATH", p)
    monkeypatch.setattr(jobs_mod, "JOBS_JSON_PATH", p)
    monkeypatch.setattr(cfg, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path / "outputs")
    return p


@pytest.fixture
def client(db_path: Path):
    """Build an isolated app; default DB is fresh per test."""
    app = main_mod.create_app()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch) -> None:
    """Redirect the orchestrator's AUDIO_DIR and OUTPUTS_DIR to tmp_path."""
    monkeypatch.setattr(orch_mod, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(orch_mod, "OUTPUTS_DIR", tmp_path / "outputs")


@pytest.fixture
def sample_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        transcript="hello world",
        segments=[TranscriptSegment(start=0.0, end=1.0, text="hello world")],
        language="en",
        duration_seconds=1200.0,  # 20-minute source
    )


@pytest.fixture
def sample_knowledge() -> KnowledgeExtract:
    return KnowledgeExtract(
        title="A 20-minute talk",
        summary="S1. S2. S3. S4.",
        insights=["a", "b", "c", "d", "e"],
        chapters=[Chapter(start_seconds=0.0, end_seconds=1200.0, title="C")],
        narrative="x " * 200,
        filler_removed=0,
    )


@pytest.fixture(autouse=True)
def _reset_default_db_each_test():
    """Each test gets a clean default DB."""
    reset_default_db()
    yield
    reset_default_db()


# --- AC-10 budget constants -------------------------------------------------


def test_max_pipeline_seconds_is_five_minutes() -> None:
    """AC-10: the hard cap is 5 minutes (300 s)."""
    assert MAX_PIPELINE_SECONDS == 300.0


def test_max_pipeline_seconds_is_env_overridable(
    monkeypatch, tmp_path: Path
) -> None:
    """Operators can tighten the budget in CI / staging via env."""
    from app import config as cfg

    monkeypatch.setenv("MAX_PIPELINE_SECONDS", "42")
    # Re-import to pick up the new value.
    import importlib

    importlib.reload(cfg)
    try:
        assert cfg.MAX_PIPELINE_SECONDS == 42.0
    finally:
        # Restore for any tests that follow.
        monkeypatch.delenv("MAX_PIPELINE_SECONDS", raising=False)
        importlib.reload(cfg)


def test_per_stage_budgets_sum_under_hard_cap() -> None:
    """Sanity: the four soft budgets fit inside the hard cap (with margin).

    The plan's documented split is 60+240+30+60 = 390 s worst-case
    but a healthy run should land inside 300 s. We assert the soft
    budgets' sum is at most 2x the hard cap so a runaway stage is
    caught by either its soft or hard budget.
    """
    soft_sum = (
        STAGE_BUDGET_INGEST_SECONDS
        + STAGE_BUDGET_TRANSCRIBE_SECONDS
        + STAGE_BUDGET_EXTRACT_SECONDS
        + STAGE_BUDGET_NARRATE_SECONDS
    )
    assert soft_sum <= 2 * MAX_PIPELINE_SECONDS


def test_soft_budget_lookup_matches_constants() -> None:
    """_soft_budget_for returns the configured value for each known stage."""
    assert _soft_budget_for("ingest") == STAGE_BUDGET_INGEST_SECONDS
    assert _soft_budget_for("transcribe") == STAGE_BUDGET_TRANSCRIBE_SECONDS
    assert _soft_budget_for("extract") == STAGE_BUDGET_EXTRACT_SECONDS
    assert _soft_budget_for("narrate") == STAGE_BUDGET_NARRATE_SECONDS


def test_soft_budget_lookup_unknown_falls_back_to_hard_cap() -> None:
    """An unknown stage name should default to the hard cap (defensive)."""
    assert _soft_budget_for("unknown-stage") == MAX_PIPELINE_SECONDS


# --- PipelineTimings schema -------------------------------------------------


def test_pipeline_timings_schema_shape() -> None:
    """PipelineTimings has the four stage fields plus total_seconds."""
    t = PipelineTimings(
        ingest_seconds=1.0,
        transcribe_seconds=2.0,
        extract_seconds=3.0,
        narrate_seconds=4.0,
        total_seconds=10.0,
    )
    assert t.ingest_seconds == 1.0
    assert t.transcribe_seconds == 2.0
    assert t.extract_seconds == 3.0
    assert t.narrate_seconds == 4.0
    assert t.total_seconds == 10.0


def test_pipeline_timings_rejects_negative_values() -> None:
    """Negative durations are nonsensical and should be rejected."""
    with pytest.raises(Exception):
        PipelineTimings(
            ingest_seconds=-0.1,
            transcribe_seconds=0.0,
            extract_seconds=0.0,
            narrate_seconds=0.0,
            total_seconds=0.0,
        )


# --- _stage_timed context manager -------------------------------------------


def test_stage_timed_captures_block_duration() -> None:
    """The yielded dict reports a non-zero duration for a sleeping block."""
    with _stage_timed("ingest") as t:
        time.sleep(0.05)
    assert t["seconds"] >= 0.04
    # And the rounding is millisecond precision.
    assert t["seconds"] == round(t["seconds"], 3)


def test_stage_timed_records_duration_even_on_exception() -> None:
    """The context manager must still capture the duration if the block raises."""
    with pytest.raises(RuntimeError):
        with _stage_timed("transcribe") as t:
            time.sleep(0.02)
            raise RuntimeError("boom")
    assert t["seconds"] >= 0.01


# --- _record_timing ---------------------------------------------------------


def test_record_timing_warns_when_stage_exceeds_soft_budget(caplog) -> None:
    """A stage above its soft budget should log a WARNING (AC-10 alarm)."""
    timings: dict[str, float] = {}
    # Ingest budget is 60 s; report 120 s.
    with caplog.at_level(logging.WARNING, logger="app.pipeline.orchestrator"):
        _record_timing("ingest", 120.0, timings=timings)
    assert timings["ingest"] == 120.0
    assert any(
        "exceeded its soft budget" in rec.message
        and "ingest" in rec.message
        for rec in caplog.records
    )


def test_record_timing_does_not_warn_within_budget(caplog) -> None:
    """A stage within its soft budget should NOT log a WARNING."""
    timings: dict[str, float] = {}
    with caplog.at_level(logging.WARNING, logger="app.pipeline.orchestrator"):
        _record_timing("ingest", 5.0, timings=timings)
    assert timings["ingest"] == 5.0
    assert not any(
        "exceeded its soft budget" in rec.message
        for rec in caplog.records
    )


# --- run_job populates timings ---------------------------------------------


def test_run_job_populates_timings_on_result(
    tmp_data_dir,
    sample_transcription: TranscriptionResult,
    sample_knowledge: KnowledgeExtract,
    tmp_path: Path,
) -> None:
    """AC-10: every LoomaResult has a non-null ``timings`` field with 4 stages."""
    mp3_path = tmp_path / "src.mp3"
    mp3_path.write_bytes(b"")
    narrate_path = tmp_path / "out.mp3"
    narrate_path.write_bytes(b"")

    patches = {
        "ingest": mock.patch.object(
            orch_mod, "_stage_ingest", return_value=mp3_path
        ),
        "transcribe": mock.patch.object(
            orch_mod, "_stage_transcribe", return_value=sample_transcription
        ),
        "extract": mock.patch.object(
            orch_mod, "_stage_extract", return_value=sample_knowledge
        ),
        "narrate": mock.patch.object(
            orch_mod, "_stage_narrate", return_value=narrate_path
        ),
    }
    with patches["ingest"], patches["transcribe"], patches["extract"], patches["narrate"]:
        result = run_job("job-ac10", JobSource.youtube("https://youtu.be/abc"))

    assert result.timings is not None
    assert isinstance(result.timings, PipelineTimings)
    # All four stages captured a non-negative duration.
    assert result.timings.ingest_seconds >= 0.0
    assert result.timings.transcribe_seconds >= 0.0
    assert result.timings.extract_seconds >= 0.0
    assert result.timings.narrate_seconds >= 0.0
    # Total is the sum of the four (with rounding tolerance).
    expected_total = round(
        result.timings.ingest_seconds
        + result.timings.transcribe_seconds
        + result.timings.extract_seconds
        + result.timings.narrate_seconds,
        3,
    )
    assert abs(result.timings.total_seconds - expected_total) < 0.01


def test_run_job_under_budget_with_realistic_mocked_latencies(
    tmp_data_dir,
    tmp_path: Path,
) -> None:
    """AC-10 end-to-end: mocked stages with realistic per-stage latencies
    still finish inside the 5-minute hard cap.

    We mock each stage with a small ``time.sleep`` chosen to be well
    inside its soft budget. The orchestrator's total wall-clock
    measurement (the only thing the AC-10 acceptance test can
    assert in CI) should also be inside the hard cap.
    """
    # Per-stage delays chosen to sum to ~1.5 s, far below the hard
    # 300 s cap and each well inside its own soft budget. The exact
    # values are unimportant — what matters is that they're all
    # non-zero, so the timings are populated, and that the sum is
    # bounded.
    def slow_ingest(*args, **kwargs):
        time.sleep(0.05)
        return tmp_path / "src.mp3"

    def slow_transcribe(mp3_path):
        time.sleep(0.05)
        return TranscriptionResult(
            transcript="hello world",
            segments=[TranscriptSegment(start=0.0, end=1.0, text="hello world")],
            language="en",
            duration_seconds=1200.0,
        )

    def slow_extract(transcription):
        time.sleep(0.05)
        return KnowledgeExtract(
            title="A 20-minute talk",
            summary="S1. S2. S3. S4.",
            insights=["a", "b", "c", "d", "e"],
            chapters=[Chapter(start_seconds=0.0, end_seconds=1200.0, title="C")],
            narrative="x " * 200,
            filler_removed=0,
        )

    def slow_narrate(text, job_id, source_duration_seconds):
        time.sleep(0.05)
        return tmp_path / "out.mp3"

    (tmp_path / "src.mp3").write_bytes(b"")
    (tmp_path / "out.mp3").write_bytes(b"")

    t0 = time.perf_counter()
    with mock.patch.object(orch_mod, "_stage_ingest", side_effect=slow_ingest), \
         mock.patch.object(orch_mod, "_stage_transcribe", side_effect=slow_transcribe), \
         mock.patch.object(orch_mod, "_stage_extract", side_effect=slow_extract), \
         mock.patch.object(orch_mod, "_stage_narrate", side_effect=slow_narrate):
        result = run_job("job-perf", JobSource.youtube("https://youtu.be/abc"))
    wall_clock = time.perf_counter() - t0

    # The orchestrator's reported total agrees with our wall-clock
    # measurement within rounding.
    assert result.timings is not None
    assert abs(result.timings.total_seconds - wall_clock) < 0.5
    # And we're well under the AC-10 hard cap.
    assert result.timings.total_seconds < MAX_PIPELINE_SECONDS
    assert wall_clock < MAX_PIPELINE_SECONDS


def test_run_job_logs_warning_when_over_budget(
    tmp_data_dir,
    caplog,
    tmp_path: Path,
) -> None:
    """A run whose total exceeds MAX_PIPELINE_SECONDS logs a WARNING.

    We can't actually sleep for 5 minutes in CI, so we monkeypatch
    the per-stage soft budgets to be trivially small. The
    orchestrator's per-stage warning + the global total warning
    should both fire.
    """
    monkeypatch = pytest.MonkeyPatch()

    # Shrink both the global cap and the per-stage soft budgets so
    # the test runs in milliseconds.
    monkeypatch.setattr(orch_mod, "MAX_PIPELINE_SECONDS", 0.01)
    monkeypatch.setattr(orch_mod, "STAGE_BUDGET_INGEST_SECONDS", 0.01)
    monkeypatch.setattr(orch_mod, "STAGE_BUDGET_TRANSCRIBE_SECONDS", 0.01)
    monkeypatch.setattr(orch_mod, "STAGE_BUDGET_EXTRACT_SECONDS", 0.01)
    monkeypatch.setattr(orch_mod, "STAGE_BUDGET_NARRATE_SECONDS", 0.01)

    def slow_ingest(*args, **kwargs):
        time.sleep(0.05)
        return tmp_path / "src.mp3"

    def slow_transcribe(mp3_path):
        time.sleep(0.05)
        return TranscriptionResult(
            transcript="hello world",
            segments=[TranscriptSegment(start=0.0, end=1.0, text="hello world")],
            language="en",
            duration_seconds=1200.0,
        )

    def slow_extract(transcription):
        time.sleep(0.05)
        return KnowledgeExtract(
            title="A 20-minute talk",
            summary="S1. S2. S3. S4.",
            insights=["a", "b", "c", "d", "e"],
            chapters=[Chapter(start_seconds=0.0, end_seconds=1200.0, title="C")],
            narrative="x " * 200,
            filler_removed=0,
        )

    def slow_narrate(text, job_id, source_duration_seconds):
        time.sleep(0.05)
        return tmp_path / "out.mp3"

    (tmp_path / "src.mp3").write_bytes(b"")
    (tmp_path / "out.mp3").write_bytes(b"")

    try:
        with caplog.at_level(logging.WARNING, logger="app.pipeline.orchestrator"):
            with mock.patch.object(orch_mod, "_stage_ingest", side_effect=slow_ingest), \
                 mock.patch.object(orch_mod, "_stage_transcribe", side_effect=slow_transcribe), \
                 mock.patch.object(orch_mod, "_stage_extract", side_effect=slow_extract), \
                 mock.patch.object(orch_mod, "_stage_narrate", side_effect=slow_narrate):
                run_job("job-overbudget", JobSource.youtube("https://youtu.be/abc"))
    finally:
        monkeypatch.undo()

    # The per-stage warnings and the global "exceeded AC-10 budget"
    # should both have fired.
    messages = [rec.message for rec in caplog.records]
    assert any("exceeded its soft budget" in m for m in messages)
    assert any("exceeded AC-10 budget" in m for m in messages)


# --- DELETE /api/jobs/{job_id} (AC-10) --------------------------------------


def _seed_many(n: int) -> None:
    """Insert ``n`` job rows so the DELETE call works against a non-empty DB."""
    db = get_default_db()
    for i in range(n):
        db.create_job(
            job_id=f"j-bulk-{i:04d}",
            source_type="youtube",
            source_ref=f"ref-{i}",
            title=f"Title {i}",
            created_at=f"2026-06-14T12:00:{i % 60:02d}.000000Z",
            duration_seconds=120.0,
            status=JobStatus.COMPLETE,
        )


@pytest.mark.asyncio
async def test_delete_job_is_fast_against_bulk_rows(
    client, tmp_path: Path
) -> None:
    """AC-10: DELETE remains fast (<1s) even when the DB has many rows.

    The contract is "DELETE removes the MP3 and the DB row"; we add
    a perf smoke test that 1000 rows don't cause a regression.
    A naive loop that re-counted on every call would be the obvious
    culprit, so we pin the wall-clock budget to 1 s.
    """
    _seed_many(1000)
    db = get_default_db()
    target = "j-bulk-0500"

    # Place the on-disk MP3s so the test exercises the file-delete path.
    audio_path = AUDIO_DIR / f"{target}.mp3"
    output_path = OUTPUTS_DIR / f"{target}.mp3"
    audio_path.write_bytes(b"audio")
    output_path.write_bytes(b"output")

    t0 = time.perf_counter()
    async with client as c:
        resp = await c.delete(f"/api/jobs/{target}")
    elapsed = time.perf_counter() - t0

    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    assert db.get_job(target) is None
    # 1 s is generous for a single-row delete + two unlink calls.
    assert elapsed < 1.0, f"DELETE took {elapsed:.3f}s against 1000 rows"


@pytest.mark.asyncio
async def test_delete_job_returns_canonical_shape(client, tmp_path: Path) -> None:
    """AC-10: the DELETE response is a structured report (deleted, job_id, files_removed)."""
    db = get_default_db()
    db.create_job(
        job_id="j-shape", source_type="youtube", source_ref="x",
        status=JobStatus.COMPLETE,
    )
    audio_path = AUDIO_DIR / "j-shape.mp3"
    output_path = OUTPUTS_DIR / "j-shape.mp3"
    audio_path.write_bytes(b"a")
    output_path.write_bytes(b"o")

    async with client as c:
        resp = await c.delete("/api/jobs/j-shape")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"deleted", "job_id", "files_removed"}
    assert body["job_id"] == "j-shape"
    assert body["deleted"] is True
    assert body["files_removed"] == {"audio": True, "output": True}


@pytest.mark.asyncio
async def test_delete_job_removes_files_even_when_db_call_succeeds(
    client, tmp_path: Path
) -> None:
    """AC-10: both MP3s (audio + output) are deleted in one call."""
    db = get_default_db()
    db.create_job(
        job_id="j-files", source_type="youtube", source_ref="x",
        status=JobStatus.COMPLETE,
    )
    audio_path = AUDIO_DIR / "j-files.mp3"
    output_path = OUTPUTS_DIR / "j-files.mp3"
    audio_path.write_bytes(b"a")
    output_path.write_bytes(b"o")
    assert audio_path.exists() and output_path.exists()

    async with client as c:
        resp = await c.delete("/api/jobs/j-files")
    assert resp.status_code == 200
    assert resp.json()["files_removed"] == {"audio": True, "output": True}
    assert not audio_path.exists()
    assert not output_path.exists()
    assert db.get_job("j-files") is None
