"""Unit tests for ``app.storage.files`` (AC-9 + AC-10 cleanup helpers).

Coverage:

* :func:`audio_path_for` / :func:`output_path_for` return the
  canonical ``<dir>/<job_id>.mp3`` shape.
* :func:`delete_job_files` removes audio + output MP3s and
  reports which were actually present.
* :func:`cleanup_orphan_files` removes old files but keeps recent
  ones. Rejects non-positive ``max_age_hours``.
* Missing-file / missing-dir cases don't crash.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.storage import files as files_mod
from app.storage.files import (
    audio_path_for,
    cleanup_orphan_files,
    delete_job_files,
    output_path_for,
)


# --- Path helpers -----------------------------------------------------------


def test_audio_path_for_shape() -> None:
    p = audio_path_for("abc-123")
    assert p.name == "abc-123.mp3"
    assert p.parent.name == "audio"


def test_output_path_for_shape() -> None:
    p = output_path_for("abc-123")
    assert p.name == "abc-123.mp3"
    assert p.parent.name == "outputs"


# --- delete_job_files -------------------------------------------------------


def test_delete_job_files_removes_existing(tmp_path: Path) -> None:
    audio = tmp_path / "j.mp3"
    output = tmp_path / "j.mp3"
    # Use separate dirs so the two files don't collide.
    audio_dir = tmp_path / "audio"
    output_dir = tmp_path / "outputs"
    audio_dir.mkdir()
    output_dir.mkdir()
    audio = audio_dir / "j.mp3"
    output = output_dir / "j.mp3"
    audio.write_bytes(b"audio")
    output.write_bytes(b"output")
    result = delete_job_files(
        "j", audio_dir=audio_dir, output_dir=output_dir,
    )
    assert result == {"audio": True, "output": True}
    assert not audio.exists()
    assert not output.exists()


def test_delete_job_files_tolerates_missing(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    output_dir = tmp_path / "outputs"
    audio_dir.mkdir()
    output_dir.mkdir()
    result = delete_job_files(
        "missing", audio_dir=audio_dir, output_dir=output_dir,
    )
    assert result == {"audio": False, "output": False}


def test_delete_job_files_partial(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    output_dir = tmp_path / "outputs"
    audio_dir.mkdir()
    output_dir.mkdir()
    (audio_dir / "j.mp3").write_bytes(b"a")
    result = delete_job_files(
        "j", audio_dir=audio_dir, output_dir=output_dir,
    )
    assert result == {"audio": True, "output": False}


# --- cleanup_orphan_files ---------------------------------------------------


def test_cleanup_orphan_removes_old_keeps_new(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    output_dir = tmp_path / "outputs"
    audio_dir.mkdir()
    output_dir.mkdir()
    # Old audio: mtime 1 hour ago.
    old = audio_dir / "old.mp3"
    old.write_bytes(b"x")
    one_hour_ago = time.time() - 3600
    import os
    os.utime(old, (one_hour_ago, one_hour_ago))
    # New audio: mtime now.
    new = audio_dir / "new.mp3"
    new.write_bytes(b"y")
    # Old output: mtime 1 hour ago.
    old_out = output_dir / "old-out.mp3"
    old_out.write_bytes(b"z")
    os.utime(old_out, (one_hour_ago, one_hour_ago))

    result = cleanup_orphan_files(
        max_age_hours=0.5,
        audio_dir=audio_dir, output_dir=output_dir,
    )
    assert result == {"audio_removed": 1, "output_removed": 1}
    assert not old.exists()
    assert new.exists()
    assert not old_out.exists()


def test_cleanup_orphan_handles_missing_dirs(tmp_path: Path) -> None:
    """Missing dirs are a no-op (the file store is lazy)."""
    missing = tmp_path / "no-such"
    result = cleanup_orphan_files(
        max_age_hours=24,
        audio_dir=missing / "audio",
        output_dir=missing / "outputs",
    )
    assert result == {"audio_removed": 0, "output_removed": 0}


def test_cleanup_orphan_rejects_non_positive_age(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_age_hours"):
        cleanup_orphan_files(
            max_age_hours=0,
            audio_dir=tmp_path, output_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="max_age_hours"):
        cleanup_orphan_files(
            max_age_hours=-1,
            audio_dir=tmp_path, output_dir=tmp_path,
        )


def test_cleanup_orphan_is_shallow(tmp_path: Path) -> None:
    """Files in subdirectories are not touched."""
    audio_dir = tmp_path / "audio"
    nested = audio_dir / "nested"
    nested.mkdir(parents=True)
    nested_file = nested / "deep.mp3"
    nested_file.write_bytes(b"x")
    # Make it ancient.
    import os
    ancient = time.time() - 86400
    os.utime(nested_file, (ancient, ancient))

    result = cleanup_orphan_files(
        max_age_hours=1,
        audio_dir=audio_dir, output_dir=audio_dir,
    )
    # Nothing was removed: the file is in a subdir.
    assert result["audio_removed"] == 0
    assert nested_file.exists()


def test_cleanup_orphan_only_targets_mp3(tmp_path: Path) -> None:
    """Non-mp3 files are not touched (defensive)."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    txt = audio_dir / "readme.txt"
    txt.write_text("hi")
    import os
    ancient = time.time() - 86400
    os.utime(txt, (ancient, ancient))
    result = cleanup_orphan_files(
        max_age_hours=1, audio_dir=audio_dir, output_dir=audio_dir,
    )
    assert result["audio_removed"] == 0
    assert txt.exists()
