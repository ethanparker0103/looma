"""Unit tests for ``app.storage.jobs`` (AC-9).

Coverage:

* CRUD — create, read, update, delete via the :class:`JobsDB` class.
* Validation — bad inputs raise :class:`ValueError`.
* Listing — most recent first, limit clamping, default = 20.
* Singleton — :func:`get_default_db` returns a process-wide instance.
* Persistence — data survives a close/re-open cycle.
* The :class:`JobRecord` Pydantic model and :class:`JobStatus` enum.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import DEFAULT_JOBS_LIMIT
from app.storage.jobs import (
    JobRecord,
    JobStatus,
    JobsDB,
    get_default_db,
    reset_default_db,
    utc_now_iso,
)


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> JobsDB:
    """Build a fresh JobsDB for the test to mutate."""
    return JobsDB(db_path=tmp_path / "jobs.json")


# --- JobRecord / JobStatus --------------------------------------------------


def test_job_status_enum_values() -> None:
    assert JobStatus.PENDING.value == "pending"
    assert JobStatus.RUNNING.value == "running"
    assert JobStatus.COMPLETE.value == "complete"
    assert JobStatus.FAILED.value == "failed"


def test_job_status_is_string_enum() -> None:
    """JobStatus is a str enum so it round-trips through JSON."""
    assert isinstance(JobStatus.COMPLETE, str)
    assert JobStatus.COMPLETE == "complete"


def test_job_record_required_fields() -> None:
    """AC-9 columns probe: every required field is present."""
    fields = set(JobRecord.model_fields.keys())
    assert {
        "id",
        "source_type",
        "source_ref",
        "title",
        "created_at",
        "duration_seconds",
        "status",
    } <= fields


def test_job_record_default_status_is_pending() -> None:
    r = JobRecord(
        id="abc", source_type="youtube", source_ref="https://x",
        created_at=utc_now_iso(),
    )
    assert r.status == JobStatus.PENDING
    assert r.title == ""
    assert r.duration_seconds == 0.0


def test_job_record_default_title_empty_string() -> None:
    r = JobRecord(
        id="abc", source_type="upload", source_ref="video.mp4",
        created_at=utc_now_iso(), status=JobStatus.RUNNING,
    )
    assert r.title == ""


# --- CRUD -------------------------------------------------------------------


def test_create_job_returns_record(db: JobsDB) -> None:
    r = db.create_job(
        job_id="j-1",
        source_type="youtube",
        source_ref="https://youtu.be/abc",
        title="Hello",
        duration_seconds=120.0,
        status=JobStatus.RUNNING,
    )
    assert isinstance(r, JobRecord)
    assert r.id == "j-1"
    assert r.source_type == "youtube"
    assert r.source_ref == "https://youtu.be/abc"
    assert r.title == "Hello"
    assert r.duration_seconds == 120.0
    assert r.status == JobStatus.RUNNING


def test_create_job_defaults(db: JobsDB) -> None:
    """Minimal call: just id, source_type, source_ref. Defaults fill the rest."""
    r = db.create_job(
        job_id="j-1",
        source_type="upload",
        source_ref="clip.mp4",
    )
    assert r.status == JobStatus.PENDING
    assert r.title == ""
    assert r.duration_seconds == 0.0
    assert r.created_at.endswith("Z")


def test_create_job_string_status_accepted(db: JobsDB) -> None:
    r = db.create_job(
        job_id="j-1", source_type="youtube", source_ref="x",
        status="complete",
    )
    assert r.status == JobStatus.COMPLETE


def test_create_job_rejects_bad_source_type(db: JobsDB) -> None:
    with pytest.raises(ValueError, match="source_type"):
        db.create_job(
            job_id="j-1", source_type="rss", source_ref="x",
        )


def test_create_job_rejects_empty_id(db: JobsDB) -> None:
    with pytest.raises(ValueError, match="job_id"):
        db.create_job(
            job_id="", source_type="youtube", source_ref="x",
        )


def test_create_job_rejects_empty_source_ref(db: JobsDB) -> None:
    with pytest.raises(ValueError, match="source_ref"):
        db.create_job(
            job_id="j-1", source_type="youtube", source_ref="",
        )


def test_create_job_rejects_invalid_status_string(db: JobsDB) -> None:
    with pytest.raises(ValueError, match="invalid status"):
        db.create_job(
            job_id="j-1", source_type="youtube", source_ref="x",
            status="not-a-status",
        )


def test_create_job_duplicate_id_raises(db: JobsDB) -> None:
    db.create_job(
        job_id="j-1", source_type="youtube", source_ref="x",
    )
    with pytest.raises(ValueError, match="already exists"):
        db.create_job(
            job_id="j-1", source_type="youtube", source_ref="y",
        )


def test_get_job_returns_record(db: JobsDB) -> None:
    db.create_job(
        job_id="j-1", source_type="youtube", source_ref="x",
        title="T", duration_seconds=42.5,
    )
    r = db.get_job("j-1")
    assert r is not None
    assert r.id == "j-1"
    assert r.title == "T"
    assert r.duration_seconds == 42.5


def test_get_job_missing_returns_none(db: JobsDB) -> None:
    assert db.get_job("nope") is None


def test_update_job_status(db: JobsDB) -> None:
    db.create_job(
        job_id="j-1", source_type="youtube", source_ref="x",
        status=JobStatus.RUNNING,
    )
    updated = db.update_job("j-1", status=JobStatus.COMPLETE)
    assert updated is not None
    assert updated.status == JobStatus.COMPLETE


def test_update_job_title_and_duration(db: JobsDB) -> None:
    db.create_job(
        job_id="j-1", source_type="youtube", source_ref="x",
    )
    updated = db.update_job(
        "j-1", title="Refined", duration_seconds=300.0,
    )
    assert updated is not None
    assert updated.title == "Refined"
    assert updated.duration_seconds == 300.0


def test_update_job_string_status(db: JobsDB) -> None:
    db.create_job(
        job_id="j-1", source_type="youtube", source_ref="x",
    )
    updated = db.update_job("j-1", status="failed")
    assert updated is not None
    assert updated.status == JobStatus.FAILED


def test_update_job_missing_returns_none(db: JobsDB) -> None:
    assert db.update_job("nope", status=JobStatus.COMPLETE) is None


def test_update_job_no_fields_returns_current(db: JobsDB) -> None:
    """A no-op update doesn't touch the file but still returns the current record."""
    db.create_job(
        job_id="j-1", source_type="youtube", source_ref="x",
        title="Original",
    )
    r = db.update_job("j-1")
    assert r is not None
    assert r.title == "Original"


def test_update_job_invalid_status_raises(db: JobsDB) -> None:
    db.create_job(
        job_id="j-1", source_type="youtube", source_ref="x",
    )
    with pytest.raises(ValueError, match="invalid status"):
        db.update_job("j-1", status="bogus")


def test_delete_job_returns_true_when_removed(db: JobsDB) -> None:
    db.create_job(
        job_id="j-1", source_type="youtube", source_ref="x",
    )
    assert db.delete_job("j-1") is True
    assert db.get_job("j-1") is None


def test_delete_job_missing_returns_false(db: JobsDB) -> None:
    assert db.delete_job("nope") is False


def test_count_jobs(db: JobsDB) -> None:
    assert db.count_jobs() == 0
    db.create_job(job_id="a", source_type="youtube", source_ref="x")
    db.create_job(job_id="b", source_type="upload", source_ref="y")
    db.create_job(job_id="c", source_type="youtube", source_ref="z")
    assert db.count_jobs() == 3
    db.delete_job("b")
    assert db.count_jobs() == 2


# --- list_jobs --------------------------------------------------------------


def test_list_jobs_empty(db: JobsDB) -> None:
    assert db.list_jobs() == []


def test_list_jobs_default_limit_is_20(db: JobsDB) -> None:
    """AC-9: ``GET /api/jobs?limit=20`` returns the most recent 20 jobs."""
    for i in range(30):
        db.create_job(
            job_id=f"j-{i:02d}", source_type="youtube", source_ref=f"x{i}",
            created_at=f"2026-06-14T12:00:{i:02d}.000000Z",
        )
    out = db.list_jobs()
    assert len(out) == DEFAULT_JOBS_LIMIT == 20


def test_list_jobs_newest_first(db: JobsDB) -> None:
    db.create_job(
        job_id="old", source_type="youtube", source_ref="x",
        created_at="2026-01-01T00:00:00.000000Z",
    )
    db.create_job(
        job_id="new", source_type="youtube", source_ref="y",
        created_at="2026-06-14T00:00:00.000000Z",
    )
    out = db.list_jobs()
    assert [r.id for r in out] == ["new", "old"]


def test_list_jobs_respects_explicit_limit(db: JobsDB) -> None:
    for i in range(5):
        db.create_job(
            job_id=f"j-{i}", source_type="youtube", source_ref=f"x{i}",
            created_at=f"2026-06-14T12:00:0{i}.000000Z",
        )
    assert len(db.list_jobs(limit=2)) == 2
    assert len(db.list_jobs(limit=10)) == 5


def test_list_jobs_clamps_lower_bound(db: JobsDB) -> None:
    db.create_job(
        job_id="j-1", source_type="youtube", source_ref="x",
    )
    # limit=0 is clamped to 1.
    assert len(db.list_jobs(limit=0)) == 1
    # limit=-5 is clamped to 1.
    assert len(db.list_jobs(limit=-5)) == 1


def test_list_jobs_clamps_upper_bound(db: JobsDB) -> None:
    for i in range(250):
        db.create_job(
            job_id=f"j-{i:03d}", source_type="youtube", source_ref=f"x{i}",
            created_at=f"2026-06-14T12:{i // 60:02d}:{i % 60:02d}.000000Z",
        )
    # limit=500 is clamped to 200 (the upper bound).
    assert len(db.list_jobs(limit=500)) == 200


def test_list_jobs_handles_non_integer_limit(db: JobsDB) -> None:
    db.create_job(
        job_id="j-1", source_type="youtube", source_ref="x",
    )
    # Non-int (e.g. None, a string) falls back to the default 20.
    out = db.list_jobs(limit=None)  # type: ignore[arg-type]
    assert len(out) == 1
    out = db.list_jobs(limit="abc")  # type: ignore[arg-type]
    assert len(out) == 1


# --- to_dict_list -----------------------------------------------------------


def test_to_dict_list_serializes_status_as_string(db: JobsDB) -> None:
    db.create_job(
        job_id="j-1", source_type="youtube", source_ref="x",
        status=JobStatus.COMPLETE,
    )
    out = db.to_dict_list(db.list_jobs())
    assert out[0]["status"] == "complete"
    assert isinstance(out[0]["status"], str)


def test_to_dict_list_default_is_all_jobs(db: JobsDB) -> None:
    for i in range(3):
        db.create_job(
            job_id=f"j-{i}", source_type="youtube", source_ref=f"x{i}",
        )
    out = db.to_dict_list()
    assert len(out) == 3


# --- Singleton --------------------------------------------------------------


def test_get_default_db_is_singleton(monkeypatch) -> None:
    """``get_default_db`` returns the same instance on repeated calls."""
    reset_default_db()
    try:
        a = get_default_db()
        b = get_default_db()
        assert a is b
    finally:
        reset_default_db()


def test_reset_default_db_drops_singleton(monkeypatch) -> None:
    reset_default_db()
    a = get_default_db()
    reset_default_db()
    b = get_default_db()
    assert a is not b
    reset_default_db()


# --- Persistence ---------------------------------------------------------


def test_data_survives_close_reopen(tmp_path: Path) -> None:
    """Jobs written survive a close() / re-open cycle."""
    p = tmp_path / "jobs.json"
    db1 = JobsDB(db_path=p)
    db1.create_job(
        job_id="persist", source_type="youtube", source_ref="x",
        title="Hello", duration_seconds=42.0,
    )
    db1.close()

    db2 = JobsDB(db_path=p)
    r = db2.get_job("persist")
    assert r is not None
    assert r.id == "persist"
    assert r.title == "Hello"
    assert r.duration_seconds == 42.0


def test_json_file_is_readable(tmp_path: Path) -> None:
    """The JSON file is valid JSON and human-readable."""
    p = tmp_path / "jobs.json"
    db = JobsDB(db_path=p)
    db.create_job(
        job_id="readable", source_type="upload", source_ref="clip.mp4",
        status=JobStatus.COMPLETE,
    )
    db.close()

    import json
    with open(p, "r") as f:
        data = json.load(f)
    assert "readable" in data
    assert data["readable"]["status"] == "complete"


# --- Helpers ----------------------------------------------------------------


def test_utc_now_iso_format() -> None:
    """Same shape as the orchestrator's helper."""
    out = utc_now_iso()
    import re
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$", out)
