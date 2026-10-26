"""Video metadata through yt-dlp. Costs no API quota."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yt_dlp
from yt_dlp.utils import DownloadError

from src.common.retry import with_retries

_log = logging.getLogger("autoclip.ytdl")
_UPCOMING_MARKERS = ("live event will begin", "premieres in", "premiere will begin")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class MetadataError(RuntimeError):
    """yt-dlp could not read the video (private, removed, members-only, sign-in wall...)."""


@dataclass(frozen=True)
class VideoMeta:
    video_id: str
    title: str
    live_status: str | None  # is_live, is_upcoming, post_live, was_live or not_live
    started_at: datetime | None
    duration_s: float | None

    @property
    def ended_at(self) -> datetime | None:
        if self.started_at is None or self.duration_s is None:
            return None
        return self.started_at + timedelta(seconds=self.duration_s)


def video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


class _YtdlLogger:
    """Routes yt-dlp output into our log instead of stdout."""

    def debug(self, msg: str) -> None:
        _log.debug(msg)

    def info(self, msg: str) -> None:
        _log.debug(msg)

    def warning(self, msg: str) -> None:
        _log.warning(msg)

    def error(self, msg: str) -> None:
        # yt-dlp also raises for every error it logs; the caller decides how loud to be.
        _log.debug(msg)


def ytdl_base_options(cookies_file: Path | None) -> dict[str, Any]:
    options: dict[str, Any] = {"quiet": True, "noprogress": True, "logger": _YtdlLogger()}
    if cookies_file is not None:
        options["cookiefile"] = str(cookies_file)
    return options


def fetch_metadata(video_id: str, cookies_file: Path | None = None) -> VideoMeta:
    try:
        info = _extract(video_id, cookies_file)
    except DownloadError as exc:
        message = _ANSI.sub("", str(exc)).removeprefix("ERROR: ").strip()
        if any(marker in message.lower() for marker in _UPCOMING_MARKERS):
            return VideoMeta(video_id, "", "is_upcoming", None, None)
        raise MetadataError(message) from exc
    started = info.get("release_timestamp") or info.get("timestamp")
    duration = info.get("duration")
    return VideoMeta(
        video_id=info.get("id") or video_id,
        title=info.get("title") or "",
        live_status=info.get("live_status"),
        started_at=datetime.fromtimestamp(started, UTC) if started else None,
        duration_s=float(duration) if duration else None,
    )


@with_retries(attempts=4)
def _extract(video_id: str, cookies_file: Path | None) -> dict[str, Any]:
    options = ytdl_base_options(cookies_file) | {
        "skip_download": True,
        # Upcoming streams have no formats yet; return their metadata instead of failing.
        "ignore_no_formats_error": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(video_url(video_id), download=False, process=False)
    if not isinstance(info, dict):
        raise MetadataError(f"yt-dlp returned no metadata for {video_id}")
    return info
