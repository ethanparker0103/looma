"""JSON-backed job history (AC-9).

Public surface:

* :class:`JobStatus` — the four state values a job can be in
  (``pending``, ``running``, ``complete``, ``failed``).
* :class:`JobRecord` — Pydantic model of a single row.
* :class:`JobsDB` — thin wrapper around a JSON file on disk.
  Use :meth:`JobsDB.create_job`, :meth:`JobsDB.update_job`,
  :meth:`JobsDB.get_job`, :meth:`JobsDB.list_jobs`, and
  :meth:`JobsDB.delete_job`.

Data model
----------
Each job is stored as one key in a JSON object, keyed by ``id``.
The JSON file looks like::

    {
      "abc123": {
        "id": "abc123",
        "source_type": "youtube",
        "source_ref": "https://...",
        "title": "My Video",
        "created_at": "2026-06-15T12:00:00.000000Z",
        "duration_seconds": 120.0,
        "status": "complete"
      },
      ...
    }

Why JSON over SQLite
--------------------
Zero dependencies, human-readable on disk, trivially inspectable
with ``cat`` or ``jq``, and fast enough for Looma's single-operator /
single-machine use case. A ``threading.Lock`` protects concurrent
reads and writes from the async event loop.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import DEFAULT_JOBS_LIMIT, JOBS_JSON_PATH

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """The four status values a job can be in (AC-9).

    Stored as a plain string in the JSON file so it shows up nicely
    in ``cat`` or ``jq`` output.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class JobRecord(BaseModel):
    """A single job from the JSON store (AC-9).

    Matches the seven AC-9 columns verbatim: ``id``,
    ``source_type``, ``source_ref``, ``title``, ``created_at``,
    ``duration_seconds``, ``status``.
    """

    id: str = Field(..., min_length=1, description="UUID hex job identifier.")
    source_type: str = Field(..., description="'youtube' or 'upload'.")
    source_ref: str = Field(..., description="Original URL or filename.")
    title: str = Field(default="", max_length=120, description="LLM-refined title.")
    created_at: str = Field(..., description="ISO-8601 UTC timestamp.")
    duration_seconds: float = Field(
        default=0.0, ge=0.0, description="Source audio duration."
    )
    status: JobStatus = Field(
        default=JobStatus.PENDING, description="Job lifecycle state."
    )


# --- Helpers ----------------------------------------------------------------


def _normalize_status(status: "JobStatus | str") -> str:
    """Coerce ``status`` into the stored string form and validate it.

    Both :class:`JobStatus` enum members and their string values
    are accepted at the API boundary. Raises :class:`ValueError`
    for an unknown value.
    """
    value = status.value if isinstance(status, JobStatus) else str(status)
    if value not in {s.value for s in JobStatus}:
        raise ValueError(f"invalid status: {value!r}")
    return value


def utc_now_iso() -> str:
    """Return the current UTC time as ISO-8601 with ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


# --- JSON store -------------------------------------------------------------


class JobsDB:
    """Thin JSON-file wrapper for the ``jobs`` store.

    Jobs are held in an in-memory ``dict`` (keyed by ``job_id``) and
    persisted to a JSON file on every write. A :class:`threading.Lock`
    guards concurrent access from the async event loop.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(
            db_path if db_path is not None else JOBS_JSON_PATH
        ).resolve()
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._loaded = False

    # --- Lifecycle ---------------------------------------------------

    @property
    def db_path(self) -> Path:
        """The on-disk path of the JSON file."""
        return self._db_path

    def _load(self) -> None:
        """Read the JSON file into ``_data`` (idempotent)."""
        if self._loaded:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        if self._db_path.exists() and self._db_path.stat().st_size > 0:
            with open(self._db_path, "r") as f:
                self._data = json.load(f)
        self._loaded = True

    def _save(self) -> None:
        """Write ``_data`` to the JSON file."""
        with open(self._db_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def close(self) -> None:
        """Clear in-memory data and reset load state (idempotent)."""
        self._data.clear()
        self._loaded = False

    # --- CRUD --------------------------------------------------------

    def create_job(
        self,
        *,
        job_id: str,
        source_type: str,
        source_ref: str,
        title: str = "",
        created_at: str | None = None,
        duration_seconds: float = 0.0,
        status: JobStatus | str = JobStatus.PENDING,
    ) -> JobRecord:
        """Insert a new job. Returns the persisted :class:`JobRecord`.

        Raises:
            ValueError: ``source_type`` is not ``"youtube"`` or ``"upload"``,
                the inputs are otherwise invalid, or a job with this
                ``job_id`` already exists.
        """
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string.")
        if source_type not in ("youtube", "upload"):
            raise ValueError(
                f"source_type must be 'youtube' or 'upload', got {source_type!r}"
            )
        if not isinstance(source_ref, str) or not source_ref:
            raise ValueError("source_ref must be a non-empty string.")
        status_value = _normalize_status(status)
        created_at = created_at or utc_now_iso()
        record = JobRecord(
            id=job_id,
            source_type=source_type,
            source_ref=source_ref,
            title=title or "",
            created_at=created_at,
            duration_seconds=float(duration_seconds),
            status=JobStatus(status_value),
        )
        raw = record.model_dump(mode="json")

        with self._lock:
            self._load()
            if job_id in self._data:
                raise ValueError(f"job_id {job_id!r} already exists")
            self._data[job_id] = raw
            self._save()

        logger.info("create_job: %s (%s)", record.id, record.status.value)
        return record

    def update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | str | None = None,
        title: str | None = None,
        duration_seconds: float | None = None,
        created_at: str | None = None,
    ) -> JobRecord | None:
        """Update one or more fields of an existing job.

        Returns the updated :class:`JobRecord`, or ``None`` if no
        job matched ``job_id``.
        """
        with self._lock:
            self._load()
            raw = self._data.get(job_id)
            if raw is None:
                return None
            if status is not None:
                raw["status"] = _normalize_status(status)
            if title is not None:
                raw["title"] = title
            if duration_seconds is not None:
                raw["duration_seconds"] = float(duration_seconds)
            if created_at is not None:
                raw["created_at"] = created_at
            self._save()

        return self._row_to_record(self._data[job_id])

    def get_job(self, job_id: str) -> JobRecord | None:
        """Fetch a job by id, or ``None`` if it doesn't exist."""
        with self._lock:
            self._load()
            raw = self._data.get(job_id)
        if raw is None:
            return None
        return self._row_to_record(raw)

    def list_jobs(self, limit: int = DEFAULT_JOBS_LIMIT) -> list[JobRecord]:
        """Return the most recent jobs, newest first.

        ``limit`` is clamped to ``[1, 200]`` so a hostile or buggy
        client can't make the server materialize the entire store.
        Defaults to :data:`app.config.DEFAULT_JOBS_LIMIT` (20).
        """
        try:
            n = int(limit)
        except (TypeError, ValueError):
            n = DEFAULT_JOBS_LIMIT
        n = max(1, min(200, n))

        with self._lock:
            self._load()
            # Sort by created_at DESC, then id DESC for stable ordering
            sorted_items = sorted(
                self._data.values(),
                key=lambda r: (r.get("created_at", ""), r.get("id", "")),
                reverse=True,
            )

        return [self._row_to_record(r) for r in sorted_items[:n]]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job. Returns ``True`` if a job was removed."""
        with self._lock:
            self._load()
            if job_id not in self._data:
                return False
            del self._data[job_id]
            self._save()
        logger.info("delete_job: removed %s", job_id)
        return True

    def count_jobs(self) -> int:
        """Return the total number of jobs in the store (for tests/admin)."""
        with self._lock:
            self._load()
            return len(self._data)

    # --- Helpers -----------------------------------------------------

    def to_dict_list(self, records: list[JobRecord] | None = None) -> list[dict]:
        """Serialize a list of records (or all jobs) as plain dicts.

        Used by the API layer when building the JSON response. The
        ``status`` is serialized as the string value (e.g.
        ``"complete"``) so JSON consumers don't need to know about
        the :class:`JobStatus` enum.
        """
        if records is None:
            records = self.list_jobs()
        out = []
        for r in records:
            out.append(
                {
                    "id": r.id,
                    "source_type": r.source_type,
                    "source_ref": r.source_ref,
                    "title": r.title,
                    "created_at": r.created_at,
                    "duration_seconds": r.duration_seconds,
                    "status": r.status.value,
                }
            )
        return out

    @staticmethod
    def _row_to_record(raw: dict[str, Any]) -> JobRecord:
        """Map a plain dict to a :class:`JobRecord`."""
        return JobRecord(
            id=raw["id"],
            source_type=raw["source_type"],
            source_ref=raw["source_ref"],
            title=raw.get("title", ""),
            created_at=raw["created_at"],
            duration_seconds=float(raw.get("duration_seconds", 0.0)),
            status=JobStatus(raw["status"]),
        )


# --- Module-level singleton ------------------------------------------------

_default_db: JobsDB | None = None


def get_default_db() -> JobsDB:
    """Return a process-wide singleton :class:`JobsDB` instance.

    The first call resolves the DB path from :data:`app.config` and
    reads the JSON file; subsequent calls reuse the cached instance.
    Tests can bypass this by constructing their own :class:`JobsDB`
    directly with a tmp path.
    """
    global _default_db
    if _default_db is None:
        _default_db = JobsDB()
    return _default_db


def reset_default_db() -> None:
    """Close + drop the cached singleton. Tests call this between cases."""
    global _default_db
    if _default_db is not None:
        _default_db.close()
        _default_db = None


__all__ = [
    "DEFAULT_JOBS_LIMIT",
    "JobRecord",
    "JobStatus",
    "JobsDB",
    "get_default_db",
    "reset_default_db",
    "utc_now_iso",
]
