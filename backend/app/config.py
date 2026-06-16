"""Application configuration constants.

Looma is single-operator / single-machine, so we read most settings from
environment variables (see `.env.example`). Constants that are part of the
AC contract — not user-tunable — live here as plain module-level
constants so tests and production code share one definition.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- AC contract constants ---------------------------------------------------

# AC-2: resulting MP3 must be under 50 MB.
MAX_AUDIO_BYTES: int = 50 * 1024 * 1024

# AC-7: narration duration must be within +/-15% of source video duration.
NARRATION_DURATION_TOLERANCE: float = 0.15

# AC-9: jobs listing default page size.
DEFAULT_JOBS_LIMIT: int = 20

# AC-10: full-pipeline wall-clock budget (5 minutes for a 20-min source).
# A run that exceeds this on a healthy 20-min source is a regression
# (likely a model down-load or a wrong Whisper size). The orchestrator
# logs a WARNING when the total exceeds this; a future run-mode flag
# can turn it into a hard timeout. Per-stage budgets are documented
# in the orchestrator module's docstring.
MAX_PIPELINE_SECONDS: float = float(
    os.environ.get("MAX_PIPELINE_SECONDS", "300")
)

# AC-10: per-stage soft budgets in seconds. Used by the performance
# tests to assert that a mocked pipeline whose stages take realistic
# per-stage latencies still fits inside the 5-minute total. The values
# are intentionally generous (well above a happy-path run) so a flaky
# CI box won't false-alarm, but tight enough to catch a stage that
# regressed by 5x.
#
# Note: the sum of these soft budgets (60+240+30+60 = 390s) is
# intentionally GREATER than ``MAX_PIPELINE_SECONDS`` (300s). The
# per-stage budgets are diagnostic warnings only — a real run is
# still contract-bound by ``MAX_PIPELINE_SECONDS`` as a hard cap.
# The plan's documented split is worst-case-per-stage, while a
# healthy run lands well inside the 300s total.
STAGE_BUDGET_INGEST_SECONDS: float = 60.0   # yt-dlp + audio normalization
STAGE_BUDGET_TRANSCRIBE_SECONDS: float = 240.0  # Whisper `small` on 20 min
STAGE_BUDGET_EXTRACT_SECONDS: float = 30.0   # LLM call + retry
STAGE_BUDGET_NARRATE_SECONDS: float = 60.0   # Edge TTS synthesis

# --- Runtime config (env-driven, with safe defaults) ------------------------

# Where JSON job store, audio, and outputs live. Resolved to an absolute path so
# the same value works for the uvicorn process and any background tasks.
DATA_DIR: Path = Path(
    os.environ.get("DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))
).resolve()

# Subdirectories under DATA_DIR. They are created on demand by the storage
# layer rather than at import time, so importing this module is side-effect
# free.
AUDIO_DIR: Path = DATA_DIR / "audio"
OUTPUTS_DIR: Path = DATA_DIR / "outputs"
JOBS_JSON_PATH: Path = DATA_DIR / "jobs.json"

# Maximum video duration we will process (90 min default). Enforced in
# ingest once metadata is known.
MAX_VIDEO_SECONDS: int = int(os.environ.get("MAX_VIDEO_SECONDS", "5400"))

# Maximum upload size in MB.
MAX_UPLOAD_MB: int = int(os.environ.get("MAX_UPLOAD_MB", "200"))

# Maximum number of pipelines that can run concurrently behind the
# async ``/api/extract/async`` endpoint. Each pipeline is CPU-bound
# (Whisper transcription) and holds ~1-2 GB peak RSS, so a hard cap
# keeps a single busy host from OOM-ing. The semaphore is created
# lazily at first use so tests can monkeypatch it before any job
# is submitted.
MAX_CONCURRENT_JOBS: int = int(os.environ.get("MAX_CONCURRENT_JOBS", "4"))

# Per-job wall-clock time budget (in seconds). The watchdog task in
# :mod:`app.main` flips any in-flight async job to ``status=timeout``
# once it has been running longer than this. Defaults to 2 hours,
# which leaves ~30 min of headroom above the worst-case 90-min
# ``MAX_VIDEO_SECONDS`` run with the heaviest Whisper model.
JOB_TIMEOUT_SECONDS: int = int(os.environ.get("JOB_TIMEOUT_SECONDS", "7200"))

# In-memory result/state TTL (in seconds). The sweeper task in
# :mod:`app.main` evicts any job whose ``updated_at`` is older than
# this. Defaults to 1 hour — long enough for the user to fetch the
# result, short enough that idle jobs don't pile up in memory.
JOB_TTL_SECONDS: int = int(os.environ.get("JOB_TTL_SECONDS", "3600"))

# Background sweeper / watchdog tick intervals (in seconds).
JOB_SWEEP_INTERVAL_SECONDS: int = int(
    os.environ.get("JOB_SWEEP_INTERVAL_SECONDS", "60")
)
JOB_WATCHDOG_INTERVAL_SECONDS: int = int(
    os.environ.get("JOB_WATCHDOG_INTERVAL_SECONDS", "10")
)
