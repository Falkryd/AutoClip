"""Where each stage reads and writes its files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.common.config import AppConfig

_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")


def validate_video_id(video_id: str) -> str:
    """Reject anything that is not an 11-character YouTube ID (also blocks path tricks)."""
    if not _VIDEO_ID.fullmatch(video_id):
        raise ValueError(f"Not a YouTube video ID: {video_id!r}")
    return video_id


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
    def chat(self) -> Path:
        return self.root / "live_chat.json"

    @property
    def audio(self) -> Path:
        return self.root / "audio.wav"

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


def state_db(cfg: AppConfig) -> Path:
    return cfg.resolve(cfg.paths.state_dir) / "autoclip.sqlite3"


def lock_path(cfg: AppConfig, name: str) -> Path:
    return cfg.resolve(cfg.paths.state_dir) / "locks" / f"{name}.lock"


def log_file(cfg: AppConfig) -> Path:
    return cfg.resolve(cfg.paths.log_dir) / "pipeline.log"
