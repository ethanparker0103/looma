"""Filesystem helpers for AC-9 + AC-10 cleanup (plan, Risk controls).

The plan calls for two helpers that the storage layer should ship:

* :func:`audio_path_for` / :func:`output_path_for` — return the
  canonical on-disk path of a job's normalized source MP3 and
  narrated MP3, respectively. These mirror the orchestrator's
  helpers so the API layer and the storage layer share one
  definition.
* :func:`delete_job_files` — remove every artifact (audio MP3,
  output MP3) for a given ``job_id``. Wired up by
  ``DELETE /api/jobs/{job_id}`` (AC-10) so a delete is symmetric:
  row + files, not just the row.
* :func:`cleanup_orphan_files` — on startup, sweep any audio or
  output files older than ``max_age_hours`` (default 24) so the
  disk doesn't fill up with abandoned runs. Plan: "Disk fill-up:
  ``cleanup_orphan_files(max_age_hours=24)`` on startup;
  ``DELETE /api/jobs/{id}`` also removes MP3s and DB row."

These helpers are intentionally minimal: they only do filesystem
work. The DB-row side of the cleanup lives in :mod:`app.storage.jobs`.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable

from ..config import AUDIO_DIR, OUTPUTS_DIR

logger = logging.getLogger(__name__)


# --- Path helpers -----------------------------------------------------------


def audio_path_for(job_id: str) -> Path:
    """Return the on-disk path of a job's normalized source MP3 (AC-2/3)."""
    return AUDIO_DIR / f"{job_id}.mp3"


def output_path_for(job_id: str) -> Path:
    """Return the on-disk path of a job's TTS narration MP3 (AC-7)."""
    return OUTPUTS_DIR / f"{job_id}.mp3"


# --- Cleanup ----------------------------------------------------------------


def _unlink_if_exists(path: Path) -> bool:
    """Remove ``path`` if it exists. Returns True on success."""
    try:
        if path.exists() and path.is_file():
            path.unlink()
            return True
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("could not unlink %s: %s", path, exc)
    return False


def delete_job_files(
    job_id: str,
    *,
    audio_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, bool]:
    """Delete every on-disk artifact associated with ``job_id``.

    Args:
        job_id: The job whose files should be removed.
        audio_dir: Override the audio directory (used by tests).
        output_dir: Override the outputs directory (used by tests).

    Returns:
        A dict with ``{"audio": bool, "output": bool}`` reporting
        which files were successfully removed. The caller (the
        API layer) is responsible for the DB-row deletion.
    """
    audio = audio_path_for(job_id) if audio_dir is None else Path(audio_dir) / f"{job_id}.mp3"
    output = output_path_for(job_id) if output_dir is None else Path(output_dir) / f"{job_id}.mp3"
    return {
        "audio": _unlink_if_exists(audio),
        "output": _unlink_if_exists(output),
    }


def _is_older_than(path: Path, max_age_seconds: float) -> bool:
    """True if ``path`` is older than ``max_age_seconds`` (by mtime)."""
    try:
        mtime = path.stat().st_mtime
    except OSError:  # pragma: no cover - defensive
        return False
    return (time.time() - mtime) > max_age_seconds


def cleanup_orphan_files(
    *,
    max_age_hours: float = 24.0,
    audio_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, int]:
    """Sweep files older than ``max_age_hours`` from audio + outputs.

    Called on app startup (AC-14 hook) and from a manual ``POST
    /api/jobs/cleanup`` route if/when we add one. The plan calls
    for a 24-hour default — old enough that an in-flight job
    won't be reaped, short enough that a 90-minute video on a
    moderately busy box doesn't accumulate to gigabytes.

    Args:
        max_age_hours: Files with mtime older than this are
            deleted. Must be > 0.
        audio_dir: Override the audio directory (used by tests).
        output_dir: Override the outputs directory (used by tests).

    Returns:
        ``{"audio_removed": n, "output_removed": m}``.
    """
    if max_age_hours <= 0:
        raise ValueError("max_age_hours must be positive.")

    max_age_seconds = max_age_hours * 3600.0
    audio_root = Path(audio_dir) if audio_dir is not None else AUDIO_DIR
    output_root = Path(output_dir) if output_dir is not None else OUTPUTS_DIR

    audio_removed = _sweep(audio_root, max_age_seconds)
    output_removed = _sweep(output_root, max_age_seconds)

    if audio_removed or output_removed:
        logger.info(
            "cleanup_orphan_files: removed %d audio, %d output (max age %.1f h)",
            audio_removed, output_removed, max_age_hours,
        )
    return {"audio_removed": audio_removed, "output_removed": output_removed}


def _sweep(directory: Path, max_age_seconds: float) -> int:
    """Remove files under ``directory`` older than ``max_age_seconds``."""
    if not directory.exists():
        return 0
    removed = 0
    for entry in _iter_mp3s(directory):
        if _is_older_than(entry, max_age_seconds):
            if _unlink_if_exists(entry):
                removed += 1
    return removed


def _iter_mp3s(directory: Path) -> Iterable[Path]:
    """Yield ``*.mp3`` files directly under ``directory`` (non-recursive).

    We keep it shallow on purpose: Looma's data dir is flat
    (``data/audio/<id>.mp3``, ``data/outputs/<id>.mp3``). A
    recursive sweep would be a footgun if a user mounted the
    data dir on top of an existing tree.
    """
    try:
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.endswith(".mp3"):
                yield Path(entry.path)
    except OSError:  # pragma: no cover - defensive
        return


__all__ = [
    "audio_path_for",
    "output_path_for",
    "cleanup_orphan_files",
    "delete_job_files",
]
