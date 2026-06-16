"""Unit tests for ``app.pipeline.ingest`` (AC-2 + AC-3).

All tests are pure unit tests — no network calls, no live ``yt-dlp``
downloads, no live ``ffmpeg`` invocations. ``yt_dlp.YoutubeDL`` and
``subprocess.run`` are patched via ``unittest.mock`` so the contracts
being verified are "the right options / argv are passed to the third
party and the right exceptions are raised" rather than "the third
party actually does the work".
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from app.pipeline import ingest
from app.pipeline.ingest import (
    AudioTooLargeError,
    DownloadFailedError,
    InvalidURLError,
    PayloadTooLargeError,
    UnsupportedMediaError,
    UnsupportedSourceError,
    _ffmpeg_convert_cmd,
    convert_upload_to_mp3,
    download_youtube,
    validate_youtube_url,
)


# --- validate_youtube_url ---------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/abcDEF12345",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
        "http://www.youtube.com/watch?v=dQw4w9WgXcQ",  # http accepted, canonicalized
    ],
)
def test_validate_youtube_url_accepts_youtube(url: str) -> None:
    result = validate_youtube_url(url)
    assert result.canonical.startswith("https://")
    assert result.video_id is not None
    assert len(result.video_id) >= 6


def test_validate_youtube_url_preserves_video_id_for_watch() -> None:
    result = validate_youtube_url("https://www.youtube.com/watch?v=abc123XYZ90")
    assert result.video_id == "abc123XYZ90"


def test_validate_youtube_url_preserves_video_id_for_short() -> None:
    result = validate_youtube_url("https://youtu.be/abc123XYZ90")
    assert result.video_id == "abc123XYZ90"


@pytest.mark.parametrize(
    "url",
    [
        "https://vimeo.com/123456789",
        "https://www.dailymotion.com/video/x7tg8e0",
        "https://soundcloud.com/foo/bar",
        "https://example.com/watch?v=abc",
    ],
)
def test_validate_youtube_url_rejects_non_youtube(url: str) -> None:
    with pytest.raises(UnsupportedSourceError):
        validate_youtube_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "not a url",
        "youtube",  # no scheme
        "://nohost",
        "https://",  # no host
        "ftp://www.youtube.com/watch?v=abc",  # wrong scheme
        "javascript:alert(1)",
    ],
)
def test_validate_youtube_url_rejects_malformed(url: str) -> None:
    with pytest.raises(InvalidURLError):
        validate_youtube_url(url)


def test_validate_youtube_url_rejects_non_string() -> None:
    with pytest.raises(InvalidURLError):
        validate_youtube_url(None)  # type: ignore[arg-type]


def test_validate_youtube_url_canonicalizes_scheme() -> None:
    result = validate_youtube_url("HTTP://WWW.YOUTUBE.COM/watch?v=abc123XYZ90")
    assert result.canonical.startswith("https://www.youtube.com/")


# --- download_youtube -------------------------------------------------------


def _make_yt_dlp_mock(
    *, expected_path: Path, file_size: int = 1024
) -> tuple[mock.MagicMock, mock.MagicMock]:
    """Build a (cls, instance) pair mocking ``yt_dlp.YoutubeDL``.

    The instance is wired as a context manager: ``__enter__`` writes a
    fake MP3 of ``file_size`` bytes at ``expected_path`` and returns
    the instance itself (so callers can use the same mock for
    assertions); ``__exit__`` is a no-op. ``ydl_cls(opts)`` returns the
    same instance every time.
    """
    ydl_instance = mock.MagicMock()

    def _enter() -> mock.MagicMock:
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_bytes(b"\x00" * file_size)
        return ydl_instance

    ydl_instance.__enter__.side_effect = _enter
    ydl_instance.__exit__.return_value = False

    ydl_cls = mock.MagicMock(return_value=ydl_instance)
    return ydl_cls, ydl_instance


def test_download_youtube_writes_mp3(tmp_path: Path) -> None:
    job_id = "job_abc"
    expected = tmp_path / f"{job_id}.mp3"

    ydl_cls, ydl = _make_yt_dlp_mock(expected_path=expected)

    with mock.patch.object(ingest.yt_dlp, "YoutubeDL", ydl_cls):
        result = download_youtube(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            job_id,
            tmp_path,
        )

    assert result == expected
    assert result.exists()
    assert result.stat().st_size > 0

    ydl.download.assert_called_once()
    (called_urls,) = ydl.download.call_args.args
    # yt_dlp.YoutubeDL.download accepts a list (or playlist) of URLs.
    assert isinstance(called_urls, list) and len(called_urls) == 1
    assert called_urls[0].startswith("https://www.youtube.com/")


def test_download_youtube_creates_output_dir(tmp_path: Path) -> None:
    job_id = "job_dir"
    expected = tmp_path / "nested" / f"{job_id}.mp3"
    ydl_cls, _ = _make_yt_dlp_mock(expected_path=expected)

    with mock.patch.object(ingest.yt_dlp, "YoutubeDL", ydl_cls):
        result = download_youtube(
            "https://youtu.be/dQw4w9WgXcQ",
            job_id,
            tmp_path / "nested",
        )

    assert result == expected


def test_download_youtube_rejects_non_youtube(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedSourceError):
        download_youtube(
            "https://vimeo.com/123456789",
            "job_x",
            tmp_path,
        )


def test_download_youtube_rejects_malformed(tmp_path: Path) -> None:
    with pytest.raises(InvalidURLError):
        download_youtube("not a url", "job_x", tmp_path)


def test_download_youtube_wraps_ytdlp_download_error(tmp_path: Path) -> None:
    ydl_cls = mock.MagicMock()
    ydl_cls.return_value.__enter__.return_value.download.side_effect = (
        ingest.yt_dlp.utils.DownloadError("network down")
    )

    with mock.patch.object(ingest.yt_dlp, "YoutubeDL", ydl_cls):
        with pytest.raises(DownloadFailedError):
            download_youtube(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "job_err",
                tmp_path,
            )


def test_download_youtube_rejects_oversize_audio(
    tmp_path: Path, monkeypatch
) -> None:
    # Lower the cap so the test doesn't have to allocate 50 MB on disk.
    monkeypatch.setattr(ingest, "MAX_AUDIO_BYTES", 1024)
    job_id = "job_big"
    expected = tmp_path / f"{job_id}.mp3"

    # The mock's __enter__ will write a 4096-byte file, larger than the
    # 1024-byte cap set above.
    ydl_cls, _ = _make_yt_dlp_mock(expected_path=expected, file_size=4096)

    with mock.patch.object(ingest.yt_dlp, "YoutubeDL", ydl_cls):
        with pytest.raises(AudioTooLargeError):
            download_youtube(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                job_id,
                tmp_path,
            )

    # AC-2 spec: "Resulting MP3 is < 50 MB" — we enforce it; the oversize
    # file should be cleaned up so we don't leak disk.
    assert not expected.exists()


def test_download_youtube_uses_ffmpeg_mp3_postprocessor() -> None:
    """AC-2: MP3 is the explicit output format via FFmpegExtractAudio."""
    from app.pipeline.ingest import _YDL_DOWNLOAD_OPTS

    pps = _YDL_DOWNLOAD_OPTS["postprocessors"]
    assert pps[0]["key"] == "FFmpegExtractAudio"
    assert pps[0]["preferredcodec"] == "mp3"


# =============================================================================
# AC-3: convert_upload_to_mp3
# =============================================================================


# --- ffmpeg argv shape ------------------------------------------------------


def test_ffmpeg_cmd_is_16khz_mono_mp3(tmp_path: Path) -> None:
    """AC-3: 16 kHz mono MP3 with ffmpeg."""
    src = tmp_path / "in.mp4"
    dst = tmp_path / "out.mp3"
    cmd = _ffmpeg_convert_cmd(src, dst)

    # Sequence matters for argv-based subprocesses.
    assert cmd[0] == "ffmpeg"
    assert "-y" in cmd  # non-interactive overwrite
    assert "-i" in cmd and cmd[cmd.index("-i") + 1] == str(src)
    assert "-vn" in cmd  # drop video
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "16000"  # 16 kHz
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"  # mono
    assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "mp3"
    assert cmd[-1] == str(dst)  # output is the last positional arg


# --- ffmpeg mock helpers ----------------------------------------------------


def _make_ffmpeg_run_mock(
    *, output_path: Path, output_size: int | None = None
) -> mock.MagicMock:
    """Build a mock ``subprocess.run`` that simulates ffmpeg writing output.

    By default, writes a 1 KB fake MP3 at ``output_path``. Pass
    ``output_size`` to write a different number of bytes (used by the
    oversize test).
    """
    size = output_size if output_size is not None else 1024

    def _run(cmd, *args, **kwargs):  # noqa: ANN001
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\x00" * size)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="", stderr=""
        )

    return mock.MagicMock(side_effect=_run)


def _make_upload_file(parent: Path, name: str = "video.mp4", size: int = 4096) -> Path:
    p = parent / name
    p.write_bytes(b"\x00" * size)
    return p


# --- convert_upload_to_mp3 --------------------------------------------------


@pytest.mark.parametrize("ext", [".mp4", ".mov", ".mkv", ".webm"])
def test_convert_upload_accepts_supported_extension(
    tmp_path: Path, ext: str
) -> None:
    upload = _make_upload_file(tmp_path, f"video{ext}")
    expected_out = tmp_path / "out" / "job_1.mp3"
    run_mock = _make_ffmpeg_run_mock(output_path=expected_out)

    with mock.patch.object(ingest.subprocess, "run", run_mock):
        result = convert_upload_to_mp3(upload, "job_1", tmp_path / "out")

    assert result == expected_out
    assert result.exists()
    assert result.stat().st_size > 0
    run_mock.assert_called_once()

    # Original upload is deleted (AC-3).
    assert not upload.exists()


@pytest.mark.parametrize(
    "name", ["foo.avi", "foo.mov.txt", "foo", "foo.MP3", "foo.m4v"]
)
def test_convert_upload_rejects_unsupported_extension(
    tmp_path: Path, name: str
) -> None:
    upload = _make_upload_file(tmp_path, name=name)
    with pytest.raises(UnsupportedMediaError):
        convert_upload_to_mp3(upload, "job_x", tmp_path / "out")
    # Original upload is NOT deleted when validation fails.
    assert upload.exists()


def test_convert_upload_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "ghost.mp4"
    with pytest.raises(UnsupportedMediaError):
        convert_upload_to_mp3(missing, "job_x", tmp_path / "out")


def test_convert_upload_rejects_oversize_upload(
    tmp_path: Path, monkeypatch
) -> None:
    # Lower the cap so the test doesn't have to allocate 200 MB on disk.
    monkeypatch.setattr(ingest, "MAX_UPLOAD_MB", 1)  # 1 MB cap
    upload = _make_upload_file(tmp_path, name="big.mp4", size=2 * 1024 * 1024)

    with pytest.raises(PayloadTooLargeError) as excinfo:
        convert_upload_to_mp3(upload, "job_x", tmp_path / "out")

    assert excinfo.value.status_code == 413
    assert excinfo.value.code == "PAYLOAD_TOO_LARGE"
    # Oversize upload is left in place — caller may want to inspect.
    assert upload.exists()


def test_convert_upload_rejects_oversize_output(
    tmp_path: Path, monkeypatch
) -> None:
    # Cap the input size high (so we don't trip the upload cap), then cap
    # the output cap low so the converted MP3 trips the AudioTooLargeError.
    monkeypatch.setattr(ingest, "MAX_UPLOAD_MB", 100)
    monkeypatch.setattr(ingest, "MAX_AUDIO_BYTES", 1024)
    upload = _make_upload_file(tmp_path, name="x.mp4", size=4096)
    expected_out = tmp_path / "out" / "job_x.mp3"
    run_mock = _make_ffmpeg_run_mock(output_path=expected_out, output_size=4096)

    with mock.patch.object(ingest.subprocess, "run", run_mock):
        with pytest.raises(AudioTooLargeError):
            convert_upload_to_mp3(upload, "job_x", tmp_path / "out")

    # Output is cleaned up; original upload is still there because
    # cleanup only happens on the success path.
    assert not expected_out.exists()
    assert upload.exists()


def test_convert_upload_wraps_ffmpeg_not_found(tmp_path: Path) -> None:
    upload = _make_upload_file(tmp_path, name="x.mp4")

    with mock.patch.object(
        ingest.subprocess,
        "run",
        side_effect=FileNotFoundError("ffmpeg missing"),
    ):
        with pytest.raises(DownloadFailedError):
            convert_upload_to_mp3(upload, "job_x", tmp_path / "out")

    # ffmpeg was never invoked, so the original upload is still there.
    assert upload.exists()


def test_convert_upload_wraps_ffmpeg_nonzero_exit(tmp_path: Path) -> None:
    upload = _make_upload_file(tmp_path, name="x.mp4")
    bad = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="Invalid data found"
    )
    with mock.patch.object(ingest.subprocess, "run", return_value=bad):
        with pytest.raises(DownloadFailedError) as excinfo:
            convert_upload_to_mp3(upload, "job_x", tmp_path / "out")
        assert "1" in str(excinfo.value)  # mentions returncode


def test_convert_upload_detects_missing_output(tmp_path: Path) -> None:
    """ffmpeg returns 0 but writes nothing -> DownloadFailedError."""
    upload = _make_upload_file(tmp_path, name="x.mp4")

    def _run_no_write(cmd, *args, **kwargs):  # noqa: ANN001
        # Return success without writing the output file.
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="", stderr=""
        )

    with mock.patch.object(
        ingest.subprocess, "run", side_effect=_run_no_write
    ):
        with pytest.raises(DownloadFailedError):
            convert_upload_to_mp3(upload, "job_x", tmp_path / "out")


def test_convert_upload_creates_output_dir(tmp_path: Path) -> None:
    upload = _make_upload_file(tmp_path, name="x.mp4")
    nested = tmp_path / "deep" / "nested" / "dir"
    expected_out = nested / "job_x.mp3"
    run_mock = _make_ffmpeg_run_mock(output_path=expected_out)

    with mock.patch.object(ingest.subprocess, "run", run_mock):
        result = convert_upload_to_mp3(upload, "job_x", nested)

    assert result == expected_out


def test_convert_upload_continues_if_unlink_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """If the post-conversion unlink fails (e.g. file already removed by
    ffmpeg's overwrite), the conversion still succeeds and we return
    the MP3 path."""
    upload = _make_upload_file(tmp_path, name="x.mp4")
    expected_out = tmp_path / "out" / "job_x.mp3"
    run_mock = _make_ffmpeg_run_mock(output_path=expected_out)

    # Force unlink to fail.
    real_unlink = Path.unlink

    def _flaky_unlink(self, *a, **kw):  # noqa: ANN001
        if self == upload:
            raise OSError("simulated permission error")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", _flaky_unlink)

    with mock.patch.object(ingest.subprocess, "run", run_mock):
        result = convert_upload_to_mp3(upload, "job_x", tmp_path / "out")

    assert result == expected_out
    assert result.exists()


# =============================================================================
# AC-3: convert_upload_to_mp3
# =============================================================================


# --- ffmpeg argv shape ------------------------------------------------------


def test_ffmpeg_cmd_is_16khz_mono_mp3(tmp_path: Path) -> None:
    """AC-3: 16 kHz mono MP3 with ffmpeg."""
    src = tmp_path / "in.mp4"
    dst = tmp_path / "out.mp3"
    cmd = _ffmpeg_convert_cmd(src, dst)

    # Sequence matters for argv-based subprocesses.
    assert cmd[0] == "ffmpeg"
    assert "-y" in cmd  # non-interactive overwrite
    assert "-i" in cmd and cmd[cmd.index("-i") + 1] == str(src)
    assert "-vn" in cmd  # drop video
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "16000"  # 16 kHz
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"  # mono
    assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "mp3"
    assert cmd[-1] == str(dst)  # output is the last positional arg


# --- ffmpeg mock helpers ----------------------------------------------------


def _make_ffmpeg_run_mock(
    *, output_path: Path, output_size: int | None = None
) -> mock.MagicMock:
    """Build a mock ``subprocess.run`` that simulates ffmpeg writing output.

    By default, writes a 1 KB fake MP3 at ``output_path``. Pass
    ``output_size`` to write a different number of bytes (used by the
    oversize test).
    """
    size = output_size if output_size is not None else 1024

    def _run(cmd, *args, **kwargs):  # noqa: ANN001
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\x00" * size)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="", stderr=""
        )

    return mock.MagicMock(side_effect=_run)


def _make_upload_file(parent: Path, name: str = "video.mp4", size: int = 4096) -> Path:
    p = parent / name
    p.write_bytes(b"\x00" * size)
    return p


# --- convert_upload_to_mp3 --------------------------------------------------


@pytest.mark.parametrize("ext", [".mp4", ".mov", ".mkv", ".webm"])
def test_convert_upload_accepts_supported_extension(
    tmp_path: Path, ext: str
) -> None:
    upload = _make_upload_file(tmp_path, f"video{ext}")
    expected_out = tmp_path / "out" / "job_1.mp3"
    run_mock = _make_ffmpeg_run_mock(output_path=expected_out)

    with mock.patch.object(ingest.subprocess, "run", run_mock):
        result = convert_upload_to_mp3(upload, "job_1", tmp_path / "out")

    assert result == expected_out
    assert result.exists()
    assert result.stat().st_size > 0
    run_mock.assert_called_once()

    # Original upload is deleted (AC-3).
    assert not upload.exists()


@pytest.mark.parametrize(
    "name", ["foo.avi", "foo.mov.txt", "foo", "foo.MP3", "foo.m4v"]
)
def test_convert_upload_rejects_unsupported_extension(
    tmp_path: Path, name: str
) -> None:
    upload = _make_upload_file(tmp_path, name=name)
    with pytest.raises(UnsupportedMediaError):
        convert_upload_to_mp3(upload, "job_x", tmp_path / "out")
    # Original upload is NOT deleted when validation fails.
    assert upload.exists()


def test_convert_upload_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "ghost.mp4"
    with pytest.raises(UnsupportedMediaError):
        convert_upload_to_mp3(missing, "job_x", tmp_path / "out")


def test_convert_upload_rejects_oversize_upload(
    tmp_path: Path, monkeypatch
) -> None:
    # Lower the cap so the test doesn't have to allocate 200 MB on disk.
    monkeypatch.setattr(ingest, "MAX_UPLOAD_MB", 1)  # 1 MB cap
    upload = _make_upload_file(tmp_path, name="big.mp4", size=2 * 1024 * 1024)

    with pytest.raises(PayloadTooLargeError) as excinfo:
        convert_upload_to_mp3(upload, "job_x", tmp_path / "out")

    assert excinfo.value.status_code == 413
    assert excinfo.value.code == "PAYLOAD_TOO_LARGE"
    # Oversize upload is left in place — caller may want to inspect.
    assert upload.exists()


def test_convert_upload_rejects_oversize_output(
    tmp_path: Path, monkeypatch
) -> None:
    # Cap the input size high (so we don't trip the upload cap), then cap
    # the output cap low so the converted MP3 trips the AudioTooLargeError.
    monkeypatch.setattr(ingest, "MAX_UPLOAD_MB", 100)
    monkeypatch.setattr(ingest, "MAX_AUDIO_BYTES", 1024)
    upload = _make_upload_file(tmp_path, name="x.mp4", size=4096)
    expected_out = tmp_path / "out" / "job_x.mp3"
    run_mock = _make_ffmpeg_run_mock(output_path=expected_out, output_size=4096)

    with mock.patch.object(ingest.subprocess, "run", run_mock):
        with pytest.raises(AudioTooLargeError):
            convert_upload_to_mp3(upload, "job_x", tmp_path / "out")

    # Output is cleaned up; original upload is still there because
    # cleanup only happens on the success path.
    assert not expected_out.exists()
    assert upload.exists()


def test_convert_upload_wraps_ffmpeg_not_found(tmp_path: Path) -> None:
    upload = _make_upload_file(tmp_path, name="x.mp4")

    with mock.patch.object(
        ingest.subprocess,
        "run",
        side_effect=FileNotFoundError("ffmpeg missing"),
    ):
        with pytest.raises(DownloadFailedError):
            convert_upload_to_mp3(upload, "job_x", tmp_path / "out")

    # ffmpeg was never invoked, so the original upload is still there.
    assert upload.exists()


def test_convert_upload_wraps_ffmpeg_nonzero_exit(tmp_path: Path) -> None:
    upload = _make_upload_file(tmp_path, name="x.mp4")
    bad = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="Invalid data found"
    )
    with mock.patch.object(ingest.subprocess, "run", return_value=bad):
        with pytest.raises(DownloadFailedError) as excinfo:
            convert_upload_to_mp3(upload, "job_x", tmp_path / "out")
        assert "1" in str(excinfo.value)  # mentions returncode


def test_convert_upload_detects_missing_output(tmp_path: Path) -> None:
    """ffmpeg returns 0 but writes nothing -> DownloadFailedError."""
    upload = _make_upload_file(tmp_path, name="x.mp4")

    def _run_no_write(cmd, *args, **kwargs):  # noqa: ANN001
        # Return success without writing the output file.
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="", stderr=""
        )

    with mock.patch.object(
        ingest.subprocess, "run", side_effect=_run_no_write
    ):
        with pytest.raises(DownloadFailedError):
            convert_upload_to_mp3(upload, "job_x", tmp_path / "out")


def test_convert_upload_creates_output_dir(tmp_path: Path) -> None:
    upload = _make_upload_file(tmp_path, name="x.mp4")
    nested = tmp_path / "deep" / "nested" / "dir"
    expected_out = nested / "job_x.mp3"
    run_mock = _make_ffmpeg_run_mock(output_path=expected_out)

    with mock.patch.object(ingest.subprocess, "run", run_mock):
        result = convert_upload_to_mp3(upload, "job_x", nested)

    assert result == expected_out


def test_convert_upload_continues_if_unlink_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """If the post-conversion unlink fails (e.g. file already removed by
    ffmpeg's overwrite), the conversion still succeeds and we return
    the MP3 path."""
    upload = _make_upload_file(tmp_path, name="x.mp4")
    expected_out = tmp_path / "out" / "job_x.mp3"
    run_mock = _make_ffmpeg_run_mock(output_path=expected_out)

    # Force unlink to fail.
    real_unlink = Path.unlink

    def _flaky_unlink(self, *a, **kw):  # noqa: ANN001
        if self == upload:
            raise OSError("simulated permission error")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", _flaky_unlink)

    with mock.patch.object(ingest.subprocess, "run", run_mock):
        result = convert_upload_to_mp3(upload, "job_x", tmp_path / "out")

    assert result == expected_out
    assert result.exists()
