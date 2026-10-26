"""Where each stage reads and writes its files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.common.config import AppConfig

_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_URL_VIDEO_ID = re.compile(r"(?:[?&]v=|youtu\.be/|/live/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def validate_video_id(video_id: str) -> str:
    """Reject anything that is not an 11-character YouTube ID (also blocks path tricks)."""
    if not _VIDEO_ID.fullmatch(video_id):
        raise ValueError(f"Not a YouTube video ID: {video_id!r}")
    return video_id


def video_id_from(text: str) -> str:
    """Accept a bare video ID or any common YouTube URL (watch, youtu.be, /live/, /shorts/)."""
    text = text.strip()
    if _VIDEO_ID.fullmatch(text):
        return text
    match = _URL_VIDEO_ID.search(text)
    if match is None:
        raise ValueError(f"Not a YouTube video ID or URL: {text!r}")
    return match.group(1)


@dataclass(frozen=True)
class Artifacts:
    """Layout of work/<video_id>/. Stages hand work to each other only through these files."""

    root: Path

    @property
    def stream_json(self) -> Path:
        return self.root / "stream.json"

    @property
    def vod(self) -> Path:
        return self.root / "vod.mp4"

    @property
    def chat_raw(self) -> Path:
        """yt-dlp's live chat replay as downloaded (JSON lines)."""
        return self.root / "live_chat.raw.jsonl"

    @property
    def chat(self) -> Path:
        """Flat list of chat messages with times relative to vod.mp4."""
        return self.root / "chat.json"

    @property
    def audio(self) -> Path:
        return self.root / "audio.wav"

    @property
    def download_manifest(self) -> Path:
        """Written last by the download stage; its presence means the stage finished."""
        return self.root / "download.json"

    @property
    def transcript(self) -> Path:
        return self.root / "transcript.json"

    @property
    def highlights(self) -> Path:
        return self.root / "highlights.json"

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def uploads(self) -> Path:
        return self.root / "uploads.json"


def artifacts(cfg: AppConfig, video_id: str) -> Artifacts:
    return Artifacts(cfg.resolve(cfg.paths.work_dir) / validate_video_id(video_id))


def samples_root(cfg: AppConfig) -> Path:
    return cfg.resolve(cfg.paths.samples_dir)


def state_db(cfg: AppConfig) -> Path:
    return cfg.resolve(cfg.paths.state_dir) / "autoclip.sqlite3"


def lock_path(cfg: AppConfig, name: str) -> Path:
    return cfg.resolve(cfg.paths.state_dir) / "locks" / f"{name}.lock"


def log_file(cfg: AppConfig) -> Path:
    return cfg.resolve(cfg.paths.log_dir) / "pipeline.log"
