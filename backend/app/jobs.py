"""In-memory job manager for the async ``/api/extract/async`` pipeline.

Why this exists
---------------
The synchronous ``/api/extract`` endpoint blocks the HTTP request for
the entire pipeline run (ingest + transcribe + extract + narrate,
typically 1-5 minutes for a long video). When the backend sits behind
Cloudflare (or any edge proxy with a request timeout) this surfaces as
a ``524`` from the proxy even though the FastAPI process is healthy.

The fix is to return ``202 Accepted`` immediately with a ``job_id`` and
let the heavy work run in a background ``asyncio`` task. The client
polls ``GET /api/jobs/{id}`` for status and ``GET /api/jobs/{id}/result``
for the final :class:`~app.models.LoomaResult`.

The partial scaffolding for this pattern already exists in the
codebase (see :class:`app.models.JobAccepted` and
:data:`app.config.MAX_CONCURRENT_JOBS`) but was never wired up. This
module finishes that work.

Concurrency
-----------
:class:`JobManager` exposes a semaphore sized to
:data:`app.config.MAX_CONCURRENT_JOBS` so a single busy host can't
OOM itself by accepting N parallel Whisper runs. The semaphore is
created lazily on first use so tests can monkey-patch
``MAX_CONCURRENT_JOBS`` before any job is submitted.

State storage
-------------
All job state lives in this process's memory (``dict[job_id, JobState]``).
There is intentionally no SQLite / file backing — a v2 that needs
multi-process state would swap this module for a Redis or KV-backed
implementation without changing the API surface. v1 is single-process
single-host; on restart, in-flight and recently completed jobs are
lost (users re-submit).

Status lifecycle
----------------
::

    queued → downloading → transcribing → done
       │          │              │
       │          │              └────► failed
       │          └───────────────────► failed
       └───────────────────────────────► failed
       │ (any non-terminal)
       └───────────────────────────────► timeout   (JOB_TIMEOUT_SECONDS)
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import JOB_TIMEOUT_SECONDS, MAX_CONCURRENT_JOBS

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Lifecycle states for an async transcription job.

    Stored as the string ``.value`` in the in-memory dict and the
    response body. A JS client can compare against the literal
    strings (``"queued"``, ``"downloading"``, ...) without importing
    the Python enum.
    """

    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    DONE = "done"
    FAILED = "failed"
    TIMEOUT = "timeout"


#: Status values that mean "still running, keep polling".
TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.DONE, JobStatus.FAILED, JobStatus.TIMEOUT}
)


@dataclass
class JobState:
    """In-memory state for a single async job.

    Lifecycle is driven by :meth:`JobManager.create` (initial), the
    background task (transitions to ``downloading`` / ``transcribing``
    / ``done`` | ``failed``), and the watchdog task (``timeout``).
    """

    id: str
    kind: str  # ``"youtube"`` or ``"upload"``
    source_ref: str  # the URL or filename, for display
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    stage_msg: str = "Queued"
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # ``deadline`` is the wall-clock time at which the watchdog should
    # mark the job as ``timeout`` and cancel the background task.
    deadline: float = field(
        default_factory=lambda: time.time() + JOB_TIMEOUT_SECONDS
    )


class JobManager:
    """Process-wide singleton holding all in-flight async jobs.

    Used as a singleton via :func:`get_job_manager` so the API layer
    and the background tasks all see the same state. The class is
    thread-safe enough for FastAPI's request-threading because all
    mutation is short-lived (``dict.__setitem__``, attribute writes)
    and the dict is never iterated-and-mutated together.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        # Semaphore caps the number of pipelines that can run
        # concurrently. Created lazily because asyncio primitives
        # need a running loop.
        self._semaphore: asyncio.Semaphore | None = None
        # Running ``asyncio.Task`` per job_id, so the watchdog can
        # cancel a stuck pipeline instead of waiting for the GIL.
        self._tasks: dict[str, asyncio.Task] = {}

    # --- CRUD --------------------------------------------------------

    def create(
        self, kind: str, source_ref: str, deadline: float | None = None
    ) -> JobState:
        """Allocate a fresh :class:`JobState` and register it.

        Args:
            kind: ``"youtube"`` or ``"upload"``. Stored verbatim.
            source_ref: The original URL (YouTube) or filename
                (upload), kept for display in the status response.
            deadline: Optional wall-clock time (seconds since epoch)
                at which the watchdog should time the job out.
                Defaults to ``now + JOB_TIMEOUT_SECONDS``.
        """
        if kind not in ("youtube", "upload"):
            raise ValueError(
                f"JobManager.create: kind must be 'youtube' or 'upload', "
                f"got {kind!r}"
            )
        job_id = uuid.uuid4().hex
        job = JobState(
            id=job_id,
            kind=kind,
            source_ref=source_ref,
            deadline=(
                deadline
                if deadline is not None
                else time.time() + JOB_TIMEOUT_SECONDS
            ),
        )
        self._jobs[job_id] = job
        logger.info(
            "JobManager.create: %s (kind=%s, source=%s)",
            job_id, kind, source_ref,
        )
        return job

    def find_by_source_ref(self, source_ref: str) -> JobState | None:
        """Return the first active job with a matching ``source_ref``.

        Used by ``_submit_async_job`` in ``app.main`` to deduplicate
        submissions for the same YouTube video: if a job for a given
        video ID is still running, the second request reuses its ID
        instead of starting a duplicate pipeline.

        Only returns jobs that are NOT in a terminal state (queued /
        downloading / transcribing) because a finished job will be
        evicted by the sweeper soon anyway.
        """
        for job in self._jobs.values():
            if job.source_ref == source_ref and job.status not in TERMINAL_STATUSES:
                return job
        return None

    def get(self, job_id: str) -> JobState | None:
        """Return the :class:`JobState` for ``job_id``, or ``None``."""
        return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs: Any) -> JobState | None:
        """Update fields on a :class:`JobState` and bump ``updated_at``.

        Returns the updated state, or ``None`` if ``job_id`` is
        unknown. ``status`` accepts either a :class:`JobStatus` enum
        member or its string value (e.g. ``"downloading"``) and
        normalizes to the enum so downstream code can rely on
        ``.value`` being available.

        Other field names are passed through to ``setattr`` so the
        watchdog / background task can set ``status``, ``progress``,
        ``stage_msg``, ``result``, ``error`` without the manager
        knowing the dataclass schema.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        for key, value in kwargs.items():
            if key == "status":
                value = self._coerce_status(value)
            setattr(job, key, value)
        job.updated_at = time.time()
        return job

    @staticmethod
    def _coerce_status(value: Any) -> JobStatus:
        """Normalize ``value`` into a :class:`JobStatus` enum member.

        Accepts:
        * A :class:`JobStatus` member (returned unchanged).
        * The string ``.value`` of a member (``"queued"``,
          ``"downloading"``, ...). Case-insensitive.

        Raises:
            ValueError: if ``value`` is not a known status string.
        """
        if isinstance(value, JobStatus):
            return value
        if isinstance(value, str):
            try:
                return JobStatus(value.lower())
            except ValueError:
                pass
        raise ValueError(
            f"JobManager.update: invalid status {value!r}; "
            f"expected JobStatus or one of "
            f"{[s.value for s in JobStatus]!r}"
        )

    def attach_task(self, job_id: str, task: "asyncio.Task") -> None:
        """Record the running ``asyncio.Task`` for ``job_id``.

        The watchdog uses this to cancel a job that has exceeded its
        ``deadline`` (see :meth:`sweep_timeouts`).
        """
        self._tasks[job_id] = task

    def detach_task(self, job_id: str) -> None:
        """Forget the task for ``job_id``. Idempotent."""
        self._tasks.pop(job_id, None)

    def list_active(self) -> list[JobState]:
        """Return all non-terminal jobs (used by the watchdog)."""
        return [
            j for j in self._jobs.values()
            if j.status not in TERMINAL_STATUSES
        ]

    def remove(self, job_id: str) -> None:
        """Drop a job from the table. Idempotent."""
        self._jobs.pop(job_id, None)
        self._tasks.pop(job_id, None)

    def sweep_expired(self, ttl_seconds: float) -> int:
        """Evict jobs whose ``updated_at`` is older than ``ttl_seconds``.

        Called by the background sweeper task once a minute. Returns
        the number of entries removed.
        """
        if ttl_seconds <= 0:
            return 0
        now = time.time()
        expired = [
            jid
            for jid, job in self._jobs.items()
            if now - job.updated_at > ttl_seconds
        ]
        for jid in expired:
            self.remove(jid)
        if expired:
            logger.info("JobManager.sweep_expired: removed %d", len(expired))
        return len(expired)

    def sweep_timeouts(self) -> list[str]:
        """Move jobs past their ``deadline`` to ``status=timeout``.

        Called by the watchdog task every few seconds. Cancels the
        running :class:`asyncio.Task` for each timed-out job so the
        pipeline doesn't keep running in the background.

        Returns the list of job_ids that were flipped to ``timeout``
        on this sweep (useful for the watchdog's log line).
        """
        now = time.time()
        flipped: list[str] = []
        for job in self.list_active():
            if now < job.deadline:
                continue
            self.update(
                job.id,
                status=JobStatus.TIMEOUT,
                stage_msg="Timed out",
                error={"code": "TIMEOUT", "msg": "Job exceeded the time budget."},
            )
            task = self._tasks.get(job.id)
            if task is not None and not task.done():
                task.cancel()
            flipped.append(job.id)
        return flipped

    # --- Concurrency primitive --------------------------------------

    def get_semaphore(self) -> asyncio.Semaphore:
        """Return the lazily-created :class:`asyncio.Semaphore`.

        Sized to :data:`app.config.MAX_CONCURRENT_JOBS` so a busy
        host can't accept more parallel Whisper runs than its RAM can
        hold.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
        return self._semaphore


# --- Module-level singleton ------------------------------------------------

_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    """Return the process-wide :class:`JobManager` singleton.

    Created lazily so importing this module is side-effect free and
    tests can call :func:`reset_job_manager` between cases.
    """
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager


def reset_job_manager() -> None:
    """Drop the cached singleton. Tests call this between cases."""
    global _manager
    _manager = None


__all__ = [
    "JOB_TIMEOUT_SECONDS",
    "JobManager",
    "JobState",
    "JobStatus",
    "TERMINAL_STATUSES",
    "get_job_manager",
    "reset_job_manager",
]
