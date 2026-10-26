"""Download one stream's VOD (or a section of it), its chat replay, and a 16 kHz audio track."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yt_dlp
from yt_dlp.utils import DownloadError

from src.common.config import AppConfig, DownloadConfig
from src.common.ffmpeg import MediaInfo, copy_cut, extract_audio, keyframe_before, probe
from src.common.files import read_json, write_json_atomic
from src.common.paths import Artifacts, artifacts, samples_root
from src.common.retry import with_retries
from src.common.state import utcnow
from src.common.timecode import compact_timecode, format_timecode, parse_timecode
from src.common.ytdl import (
    NOT_READY_STATUSES,
    VideoMeta,
    YtdlEnv,
    fetch_metadata,
    video_url,
    ytdl_base_options,
    ytdl_env,
    ytdl_error_message,
)
from src.download.chat import chat_document, clip_to_window, parse_chat_lines

_log = logging.getLogger("autoclip.download")

AUDIO_SAMPLE_RATE = 16_000


class DownloadStageError(RuntimeError):
    """The VOD can't be downloaded, or what came back is unusable."""


@dataclass(frozen=True)
class Section:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def label(self) -> str:
        return f"{format_timecode(self.start_s)}-{format_timecode(self.end_s)}"


def parse_section(text: str) -> Section:
    """'2:33:00-3:33:00' -> Section(9180, 12780)."""
    start, separator, end = text.partition("-")
    if not separator:
        raise ValueError(f"Section must look like START-END, e.g. 2:33:00-3:33:00: {text!r}")
    section = Section(parse_timecode(start), parse_timecode(end))
    if section.end_s <= section.start_s:
        raise ValueError(f"Section ends before it starts: {text!r}")
    return section


def sample_dir(cfg: AppConfig, video_id: str, section: Section) -> Path:
    start, end = compact_timecode(section.start_s), compact_timecode(section.end_s)
    return samples_root(cfg) / f"{video_id}_{start}-{end}"


def is_complete(a: Artifacts) -> bool:
    return all(path.is_file() for path in (a.download_manifest, a.vod, a.chat, a.audio))


def run_download(
    cfg: AppConfig,
    video_id: str,
    a: Artifacts,
    section: Section | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Each step skips when its output already exists (unless force), so a rerun resumes."""
    env = ytdl_env(cfg)
    meta = fetch_metadata(video_id, env)
    if meta.live_status in NOT_READY_STATUSES:
        raise DownloadStageError(f"{video_id} has no finished VOD yet ({meta.live_status})")
    if section and meta.duration_s and section.end_s > meta.duration_s:
        raise DownloadStageError(
            f"Section {section.label()} runs past the end of the stream "
            f"({format_timecode(meta.duration_s)})"
        )
    a.root.mkdir(parents=True, exist_ok=True)

    # A section is cut locally from the full VOD in work/<id>/. Letting yt-dlp fetch the
    # range itself hands the job to FFmpeg, which YouTube stalls on long streams; the native
    # downloader is fast, and the full VOD is then already in place for the real pipeline.
    full_vod = a.vod if section is None else artifacts(cfg, video_id).vod
    if not full_vod.is_file() or (force and section is None):
        _log.info("Downloading full VOD of %s (%s)", video_id, meta.title)
        download_vod(video_id, full_vod, cfg.download, env)
    else:
        _log.info("Keeping existing %s", full_vod)
    offset_s = 0.0  # where vod.mp4 starts in the full stream
    if section:
        if force or not a.vod.is_file():
            _log.info("Cutting %s out of the full VOD", section.label())
            copy_cut(full_vod, a.vod, section.start_s, section.duration_s)
        # A stream-copy cut starts on the keyframe at or before the requested start (seconds
        # earlier on YouTube VODs). Chat and captions must line up with that, not the request.
        offset_s = keyframe_before(full_vod, section.start_s)
        _log.info("Sample starts at %.3f s in the stream", offset_s)
    media = probe(a.vod)
    if not media.has_audio:
        raise DownloadStageError(f"{a.vod} has no audio track")
    if section and abs(media.duration_s - section.duration_s) > 10:
        _log.warning(
            "vod.mp4 is %.0f s long but the section is %.0f s", media.duration_s, section.duration_s
        )

    if force or not a.chat.is_file():
        write_json_atomic(
            a.chat, fetch_chat(video_id, a, offset_s, media.duration_s, env, force=force)
        )
    chat = read_json(a.chat)

    if force or not a.audio.is_file():
        _log.info("Extracting 16 kHz mono audio")
        extract_audio(a.vod, a.audio, AUDIO_SAMPLE_RATE)

    manifest = _manifest(meta, a, section, offset_s, media, chat)
    write_json_atomic(a.download_manifest, manifest)  # written last: marks the stage done
    return manifest


def download_vod(video_id: str, dest: Path, dl: DownloadConfig, env: YtdlEnv) -> None:
    """Full VOD to `dest` (an .mp4 path) with yt-dlp's own chunked downloader."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    options = ytdl_base_options(env) | {
        "format": dl.format,
        "merge_output_format": "mp4",
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
        "outtmpl": {"default": str(dest.with_suffix(".%(ext)s"))},
        "retries": dl.retries,
        "fragment_retries": dl.retries,
        "concurrent_fragment_downloads": 4,
        "overwrites": True,
        "progress_hooks": [_ProgressLog()],
    }
    try:
        _run_ytdl(options, video_id)
    except DownloadError as exc:
        raise DownloadStageError(f"VOD download failed: {ytdl_error_message(exc)}") from exc
    if not dest.is_file():
        raise DownloadStageError(f"yt-dlp finished but {dest} is missing")


def fetch_chat(
    video_id: str,
    a: Artifacts,
    offset_s: float,
    vod_duration_s: float,
    env: YtdlEnv,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """chat.json for vod.mp4, which starts offset_s seconds into the stream."""
    offset = offset_s
    available = a.chat_raw.is_file() and not force
    if not available:
        available = _download_chat_raw(video_id, a, env)
    if not available:
        _log.warning("%s has no live chat replay; highlights will score without chat", video_id)
        return chat_document(
            video_id, [], available=False, offset_s=offset, duration_s=vod_duration_s,
            skipped_lines=0,
        )  # fmt: skip
    with a.chat_raw.open(encoding="utf-8") as fh:
        messages, bad_lines = parse_chat_lines(fh)
    if bad_lines:
        _log.warning("Skipped %d unreadable chat lines", bad_lines)
    messages = clip_to_window(messages, offset, offset + vod_duration_s)
    _log.info("Chat: %d messages in this part of the stream", len(messages))
    return chat_document(
        video_id, messages, available=True, offset_s=offset, duration_s=vod_duration_s,
        skipped_lines=bad_lines,
    )  # fmt: skip


def _download_chat_raw(video_id: str, a: Artifacts, env: YtdlEnv) -> bool:
    """Saves the raw replay to live_chat.raw.jsonl. False when the stream has no replay."""
    options = ytdl_base_options(env) | {
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": ["live_chat"],
        "outtmpl": {"default": str(a.root / "chat_download.%(ext)s")},
    }
    _log.info("Downloading live chat replay (a long stream takes a few minutes)")
    try:
        info = _run_ytdl(options, video_id)
    except DownloadError as exc:
        raise DownloadStageError(f"Chat download failed: {ytdl_error_message(exc)}") from exc
    subtitle = (info.get("requested_subtitles") or {}).get("live_chat")
    if not subtitle:
        return False
    path = Path(subtitle.get("filepath") or a.root / "chat_download.live_chat.json")
    if not path.is_file():
        raise DownloadStageError(f"yt-dlp reported chat at {path} but the file is missing")
    os.replace(path, a.chat_raw)
    return True


@with_retries(attempts=3)
def _run_ytdl(options: dict[str, Any], video_id: str) -> dict[str, Any]:
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(video_url(video_id), download=True)
    return info if isinstance(info, dict) else {}


class _ProgressLog:
    """Logs progress every 10 % instead of yt-dlp's per-chunk output."""

    def __init__(self) -> None:
        self._next_pct = 10

    def __call__(self, status: dict[str, Any]) -> None:
        name = Path(status.get("filename") or "").name
        if status.get("status") == "finished":
            _log.info("Downloaded %s", name)
            self._next_pct = 10
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        if status.get("status") != "downloading" or not total:
            return
        pct = 100 * (status.get("downloaded_bytes") or 0) / total
        if pct >= self._next_pct:
            _log.info("%s: %d%% of %.0f MB", name, pct, total / 1e6)
            self._next_pct = (int(pct) // 10 + 1) * 10


def _manifest(
    meta: VideoMeta,
    a: Artifacts,
    section: Section | None,
    offset_s: float,
    media: MediaInfo,
    chat: dict[str, Any],
) -> dict[str, Any]:
    return {
        "video_id": meta.video_id,
        "title": meta.title,
        "url": video_url(meta.video_id),
        "live_status": meta.live_status,
        "stream_duration_s": meta.duration_s,
        "offset_s": offset_s,  # where vod.mp4 starts in the stream; captions and chat use it
        "section": (
            {"start_s": section.start_s, "end_s": section.end_s, "label": section.label()}
            if section
            else None
        ),
        "vod": {
            "file": a.vod.name,
            "duration_s": media.duration_s,
            "width": media.width,
            "height": media.height,
            "fps": media.fps,
            "video_start_s": media.video_start_s,
            "audio_start_s": media.audio_start_s,
        },
        "chat": {
            "file": a.chat.name,
            "available": chat["available"],
            "messages": chat["message_count"],
        },
        "audio": {"file": a.audio.name, "sample_rate": AUDIO_SAMPLE_RATE},
        "completed_at": utcnow().isoformat(timespec="seconds"),
    }
