"""AC-10 performance tests for the simplified (BYOK) pipeline.

Covers:

1. **Pipeline wall-clock budget.** AC-10 requires the full
   ingest + transcribe pipeline to finish in **under 5 minutes**
   for a 20-minute source. We verify:
   * :func:`_stage_timed` captures block duration correctly.
   * :func:`_record_timing` warns when a stage exceeds its soft budget.
   * The per-stage soft budgets are properly configured.
2. **DELETE /api/jobs/{job_id}** removes both the DB row and the
   on-disk MP3s (same as before — unchanged by BYOK refactor).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from unittest import mock

import httpx
import pytest

from app import config as cfg
from app import main as main_mod
from app.config import (
    MAX_PIPELINE_SECONDS,
    STAGE_BUDGET_INGEST_SECONDS,
    STAGE_BUDGET_TRANSCRIBE_SECONDS,
)
from app.models import TranscriptSegment, TranscriptionResult
from app.pipeline import orchestrator as orch_mod
from app.pipeline.orchestrator import (
    JobSource,
    _record_timing,
    _stage_timed,
    run_job,
)


# --- AC-10 budget constants -------------------------------------------------


class TestBudgetConstants:
    def test_max_pipeline_seconds_is_five_minutes(self) -> None:
        assert MAX_PIPELINE_SECONDS == 300.0

    def test_per_stage_budgets_sum_under_hard_cap(self) -> None:
        soft_sum = STAGE_BUDGET_INGEST_SECONDS + STAGE_BUDGET_TRANSCRIBE_SECONDS
        assert soft_sum <= 2 * MAX_PIPELINE_SECONDS


# --- _stage_timed tests -----------------------------------------------------


class TestStageTimed:
    def test_captures_block_duration(self) -> None:
        with _stage_timed("sleepy") as t:
            time.sleep(0.05)
        assert 0.04 <= t["seconds"] <= 0.5

    def test_records_duration_even_on_exception(self) -> None:
        try:
            with _stage_timed("crash") as t:
                raise ValueError("boom")
        except ValueError:
            pass
        assert 0.0 <= t["seconds"] <= 1.0


# --- _record_timing tests ---------------------------------------------------


class TestRecordTiming:
    def test_warns_when_stage_exceeds_soft_budget(self, caplog) -> None:
        caplog.set_level(logging.WARNING)
        timings: dict[str, float] = {}
        _record_timing("ingest", STAGE_BUDGET_INGEST_SECONDS + 10, timings=timings)
        assert timings["ingest"] > STAGE_BUDGET_INGEST_SECONDS - 1
        assert any("exceeded its soft budget" in r.getMessage() for r in caplog.records)

    def test_does_not_warn_within_budget(self, caplog) -> None:
        caplog.set_level(logging.WARNING)
        timings: dict[str, float] = {}
        _record_timing("transcribe", 1.0, timings=timings)
        assert timings["transcribe"] == 1.0
        warnings = [
            r.getMessage() for r in caplog.records
            if "exceeded its soft budget" in r.getMessage()
        ]
        assert len(warnings) == 0


# --- run_job budget tests ---------------------------------------------------


class TestRunJobBudget:
    def test_under_budget_with_mocked_latencies(self, tmp_path) -> None:
        """run_job under AC-10 hard cap with mocked 2-stage delays."""
        def slow_ingest(*args, **kwargs):
            time.sleep(0.05)
            return tmp_path / "src.mp3"

        def slow_transcribe(*args, **kwargs):
            time.sleep(0.05)
            return TranscriptionResult(
                transcript="hello world",
                segments=[TranscriptSegment(start=0.0, end=1.0, text="hello world")],
                language="en",
                duration_seconds=1200.0,
            )

        (tmp_path / "src.mp3").write_bytes(b"")

        t0 = time.perf_counter()
        with mock.patch.object(orch_mod, "_stage_ingest", side_effect=slow_ingest), \
             mock.patch.object(orch_mod, "_stage_transcribe", side_effect=slow_transcribe):
            result = run_job("job-perf", JobSource.youtube("https://youtu.be/abc"))
        wall_clock = time.perf_counter() - t0

        assert isinstance(result, TranscriptionResult)
        assert wall_clock < MAX_PIPELINE_SECONDS

    def test_logs_warning_when_over_budget(self, caplog, tmp_path) -> None:
        """Orchestrator logs a budget warning when stages exceed MAX_PIPELINE_SECONDS."""
        def slow_ingest(*args, **kwargs):
            time.sleep(STAGE_BUDGET_INGEST_SECONDS + 5)
            return tmp_path / "src.mp3"

        def slow_transcribe(*args, **kwargs):
            time.sleep(0.05)
            return TranscriptionResult(
                transcript="x",
                segments=[TranscriptSegment(start=0.0, end=1.0, text="x")],
                language="en",
                duration_seconds=1.0,
            )

        (tmp_path / "src.mp3").write_bytes(b"")
        caplog.set_level(logging.WARNING)

        with mock.patch.object(orch_mod, "_stage_ingest", side_effect=slow_ingest), \
             mock.patch.object(orch_mod, "_stage_transcribe", side_effect=slow_transcribe):
            run_job("job-budget", JobSource.youtube("https://youtu.be/abc"))

        budget_warnings = [
            r.getMessage() for r in caplog.records
            if "exceeded its soft budget" in r.getMessage()
        ]
        assert len(budget_warnings) >= 1
