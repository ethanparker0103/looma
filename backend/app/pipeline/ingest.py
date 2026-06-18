"""Audio ingestion: YouTube URL downloads and upload conversion (AC-2, AC-3).

Public entry points:

* :func:`download_youtube` — fetch a YouTube video's audio track as MP3 into
  ``data/audio/<job_id>.mp3`` (AC-2).
* :func:`convert_upload_to_mp3` — convert an uploaded video to a
  normalized 16 kHz mono MP3 (AC-3).

Domain-specific exceptions (subclasses of :class:`IngestError`) carry an
HTTP status hint and a machine-readable error code (matching the API
contract in `app/models.py`) and are mapped to HTTP 4xx/5xx responses by
the API layer per AC-11.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yt_dlp

from ..config import MAX_AUDIO_BYTES, MAX_UPLOAD_MB


# --- Exceptions -------------------------------------------------------------


class IngestError(Exception):
    """Base class for ingest-layer failures.

    Subclasses carry an HTTP status hint and a machine-readable error code
    (matching the API contract in `app/models.py`).
    """

    status_code: int = 500
    code: str = "INGEST_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidURLError(IngestError):
    status_code = 400
    code = "INVALID_URL"


class UnsupportedSourceError(IngestError):
    status_code = 400
    code = "UNSUPPORTED_SOURCE"


class DownloadFailedError(IngestError):
    status_code = 500
    code = "DOWNLOAD_FAILED"


class AudioTooLargeError(IngestError):
    status_code = 413
    code = "PAYLOAD_TOO_LARGE"


class UnsupportedMediaError(IngestError):
    status_code = 415
    code = "UNSUPPORTED_MEDIA"


class PayloadTooLargeError(IngestError):
    status_code = 413
    code = "PAYLOAD_TOO_LARGE"


# --- URL validation ---------------------------------------------------------

# Hosts that we consider YouTube. ``youtube.com`` covers the desktop site,
# ``m.youtube.com`` is the mobile site, ``youtu.be`` is the short-URL form, and
# ``youtube-nocookie.com`` is the privacy-enhanced embed domain. We accept
# any of these as "YouTube" — anything else raises UnsupportedSourceError.
_YOUTUBE_HOSTS: frozenset[str] = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
    }
)

# A loose check for "looks like a URL": non-empty, has at least one dot in
# the host, no whitespace. This is intentionally permissive — strict parse
# failures become InvalidURLError.
_URLISH = re.compile(
    r"^https?://[^\s/?#]+\.[^\s/?#]+(?:[/?#][^\s]*)?$", re.IGNORECASE
)


@dataclass(frozen=True)
class YouTubeURL:
    """A validated YouTube URL.

    Attributes:
        original: The URL exactly as it was passed in (after trimming).
        canonical: The URL with scheme normalized to ``https`` and the
            host lower-cased.
        video_id: The YouTube video id when we can extract it from the
            query string or the path (for ``youtu.be`` links). May be
            ``None`` for unusual URLs such as ``/playlist?list=...``
            which we still accept as YouTube but cannot extract a video
            id from.
    """

    original: str
    canonical: str
    video_id: str | None


def validate_youtube_url(url: str) -> YouTubeURL:
    """Validate that ``url`` is a YouTube URL.

    Raises:
        InvalidURLError: The string is empty, has no scheme, contains
            whitespace, or otherwise cannot be parsed as an http(s) URL.
        UnsupportedSourceError: The URL parses but its host is not a
            YouTube host.
    """
    if not isinstance(url, str):  # type: ignore[unreachable]
        raise InvalidURLError("URL must be a string.")

    cleaned = url.strip()
    if not cleaned:
        raise InvalidURLError("URL is empty.")

    if not _URLISH.match(cleaned):
        # Reject obvious non-URLs ("not a url", "youtube", "//foo/bar")
        # before we even hand them to urlparse.
        raise InvalidURLError(f"Not a valid http(s) URL: {cleaned!r}")

    try:
        parsed = urlparse(cleaned)
    except ValueError as exc:  # pragma: no cover - urlparse almost never raises
        raise InvalidURLError(f"Could not parse URL: {cleaned!r}") from exc

    if parsed.scheme not in ("http", "https"):
        raise InvalidURLError(
            f"URL scheme must be http or https, got {parsed.scheme!r}"
        )

    host = (parsed.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS:
        raise UnsupportedSourceError(
            f"Host {host!r} is not a YouTube domain. "
            "Only youtube.com and youtu.be URLs are supported in v1."
        )

    canonical = f"https://{host}{parsed.path}"
    if parsed.query:
        canonical = f"{canonical}?{parsed.query}"
    if parsed.fragment:
        canonical = f"{canonical}#{parsed.fragment}"

    video_id = _extract_video_id(parsed)
    return YouTubeURL(original=cleaned, canonical=canonical, video_id=video_id)


def _extract_video_id(parsed: Any) -> str | None:
    """Best-effort extraction of the YouTube video id from a parsed URL.

    Handles the three URL shapes Looma will see in practice:
    * ``https://www.youtube.com/watch?v=ID``
    * ``https://youtu.be/ID``
    * ``https://www.youtube.com/shorts/ID``
    """
    from urllib.parse import parse_qs

    host = (parsed.hostname or "").lower()
    path = parsed.path or ""

    if host.endswith("youtu.be"):
        # Path is "/<id>".
        seg = path.lstrip("/").split("/", 1)[0]
        return seg or None

    if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        # /watch?v=ID
        qs = parse_qs(parsed.query or "")
        if "v" in qs and qs["v"]:
            return qs["v"][0]
        # /shorts/ID or /embed/ID or /live/ID
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
            return parts[1]

    return None


# --- Download ---------------------------------------------------------------


# A bounded postprocessor chain: extract audio, re-encode to MP3 at 192k
# (good enough for speech at small file size), and let ffmpeg place the
# final file at <output_dir>/<job_id>.mp3.
_YDL_DOWNLOAD_OPTS: dict[str, Any] = {
    "format": "bestaudio/best",
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "extractor_retries": 5,
    "retries": 10,
    "fragment_retries": 10,
    "extractor_args": {
        "youtube": {
            "player_client": ["android", "web"],
            "skip": ["webpage", "dash", "hls"],
        },
    },
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ],
}


def download_youtube(
    url: str,
    job_id: str,
    output_dir: Path | str,
) -> Path:
    """Download a YouTube video's audio as MP3 to ``output_dir/<job_id>.mp3``.

    Args:
        url: The user-supplied YouTube URL. Will be validated first.
        job_id: A unique identifier for this job; used as the output
            filename. The caller is responsible for ensuring uniqueness.
        output_dir: Directory the MP3 will be written to. Created if it
            does not exist.

    Returns:
        The :class:`pathlib.Path` of the resulting MP3 file.

    Raises:
        InvalidURLError, UnsupportedSourceError, DownloadFailedError,
        AudioTooLargeError: See the exception classes above.
    """
    yt = validate_youtube_url(url)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    final_path = out / f"{job_id}.mp3"

    # yt-dlp writes the intermediate download to ``outtmpl`` relative to
    # the cwd, so we pin both the cwd (via outtmpl's parent directory)
    # and the template to our output directory. ``outtmpl`` must be a
    # string (not a list) for ``_outtmpl_expandpath`` in yt-dlp
    # 2024.10.07 — passing a list raises
    # ``AttributeError: 'list' object has no attribute 'replace'``.
    opts: dict[str, Any] = {
        **_YDL_DOWNLOAD_OPTS,
        "outtmpl": str(out / f"{job_id}.%(ext)s"),
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([yt.canonical])
    except yt_dlp.utils.DownloadError as exc:
        raise DownloadFailedError(
            f"yt-dlp failed to download {yt.canonical!r}: {exc}"
        ) from exc
    except yt_dlp.utils.UnsupportedError as exc:  # pragma: no cover
        # Some regions / age-restricted videos raise this.
        raise UnsupportedSourceError(
            f"yt-dlp reports this URL is unsupported: {exc}"
        ) from exc

    if not final_path.exists():
        # yt-dlp will sometimes honor an existing filename (e.g. when the
        # video id is what we asked for) and leave a differently-named
        # file in the directory. Fall back to glob-matching any MP3.
        candidates = sorted(out.glob(f"{job_id}*.mp3"))
        if not candidates:
            raise DownloadFailedError(
                f"yt-dlp reported success but no MP3 found at {final_path}"
            )
        # If we have a file with a different name, move it to the canonical
        # location so downstream stages can rely on the contract.
        if candidates[0] != final_path:
            candidates[0].rename(final_path)

    size = final_path.stat().st_size
    if size > MAX_AUDIO_BYTES:
        # Clean up the oversize file so we don't leak disk.
        try:
            final_path.unlink()
        except OSError:  # pragma: no cover
            pass
        raise AudioTooLargeError(
            f"Downloaded audio is {size} bytes, which exceeds the "
            f"{MAX_AUDIO_BYTES}-byte cap."
        )

    return final_path


# --- Upload conversion (AC-3) ----------------------------------------------


# Extensions that the upload conversion step will accept. Anything else
# raises UnsupportedMediaError (HTTP 415). Compared case-insensitively
# against the file's suffix.
SUPPORTED_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".mkv", ".webm"}
)


def _ffmpeg_convert_cmd(src: Path, dst: Path) -> list[str]:
    """Build the ffmpeg argv for normalizing a video to 16 kHz mono MP3.

    Pulled out as a helper so tests can assert on the exact command
    shape (16 kHz / mono / MP3) without invoking ffmpeg.
    """
    return [
        "ffmpeg",
        "-y",  # overwrite output without prompting
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(src),
        "-vn",  # discard video stream
        "-ar", "16000",  # 16 kHz sample rate (AC-3)
        "-ac", "1",  # mono (AC-3)
        "-b:a", "192k",  # 192 kbps MP3 bitrate
        "-f", "mp3",
        str(dst),
    ]


def convert_upload_to_mp3(
    upload_path: Path | str,
    job_id: str,
    output_dir: Path | str,
) -> Path:
    """Convert an uploaded video to 16 kHz mono MP3 (AC-3).

    Args:
        upload_path: Path to the uploaded file on disk. The file is
            deleted from disk after a successful conversion (AC-3:
            "Original upload is deleted after conversion"). Callers
            that need to retain the original must copy it first.
        job_id: Unique identifier for this job; used as the output
            filename.
        output_dir: Directory the MP3 will be written to. Created if
            it does not exist.

    Returns:
        The :class:`pathlib.Path` of the resulting MP3 file.

    Raises:
        UnsupportedMediaError: The file extension is not in
            :data:`SUPPORTED_VIDEO_EXTENSIONS` (HTTP 415) or the file
            does not exist.
        PayloadTooLargeError: The upload exceeds
            ``MAX_UPLOAD_MB * 1024 * 1024`` bytes (HTTP 413).
        DownloadFailedError: ffmpeg is missing, fails, or does not
            produce the expected output (HTTP 500).
        AudioTooLargeError: The resulting MP3 exceeds
            :data:`MAX_AUDIO_BYTES` (50 MB) — the file is unlinked
            before raising.
    """
    src = Path(upload_path)
    out = Path(output_dir)

    # --- 1. Extension check ----------------------------------------------
    ext = src.suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise UnsupportedMediaError(
            f"Upload extension {ext!r} is not supported. "
            f"Use one of: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}."
        )

    if not src.exists() or not src.is_file():
        raise UnsupportedMediaError(
            f"Upload file not found at {src!s}."
        )

    # --- 2. Size check ---------------------------------------------------
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    size = src.stat().st_size
    if size > max_bytes:
        raise PayloadTooLargeError(
            f"Upload is {size} bytes, which exceeds the "
            f"{MAX_UPLOAD_MB}-MB cap ({max_bytes} bytes)."
        )

    # --- 3. ffmpeg conversion -------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    final_path = out / f"{job_id}.mp3"
    cmd = _ffmpeg_convert_cmd(src, final_path)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as exc:
        # ffmpeg is not on PATH. AC-14 catches this at startup, but the
        # user could have uninstalled ffmpeg between boot and request.
        raise DownloadFailedError(
            "ffmpeg was not found on PATH. "
            "Install it with `apt-get install -y ffmpeg`."
        ) from exc
    except OSError as exc:  # pragma: no cover - defensive
        raise DownloadFailedError(
            f"Could not invoke ffmpeg for {src.name}: {exc}"
        ) from exc

    if result.returncode != 0:
        # Capture a short tail of stderr for diagnostics without leaking
        # the entire ffmpeg log into the error message.
        stderr_tail = (result.stderr or "").strip().splitlines()[-3:]
        stderr_hint = ("\n".join(stderr_tail)).strip() or "<no stderr>"
        raise DownloadFailedError(
            f"ffmpeg exited with status {result.returncode} for "
            f"{src.name}: {stderr_hint}"
        )

    # --- 4. Output validation ------------------------------------------
    if not final_path.exists():
        raise DownloadFailedError(
            f"ffmpeg reported success but no MP3 was found at "
            f"{final_path!s}."
        )

    out_size = final_path.stat().st_size
    if out_size > MAX_AUDIO_BYTES:
        try:
            final_path.unlink()
        except OSError:  # pragma: no cover
            pass
        raise AudioTooLargeError(
            f"Converted audio is {out_size} bytes, which exceeds the "
            f"{MAX_AUDIO_BYTES}-byte cap."
        )

    # --- 5. Delete the original upload (AC-3) ---------------------------
    try:
        src.unlink()
    except OSError as exc:
        # The conversion succeeded; we do not fail the request just
        # because cleanup is stuck. The orphan file will be reaped by
        # cleanup_orphan_files in a later round. Surfacing the error as
        # a warning keeps the user-facing result usable.
        import logging

        logging.getLogger(__name__).warning(
            "Could not delete original upload %s after conversion: %s",
            src, exc,
        )

    return final_path
