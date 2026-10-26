"""FFmpeg and ffprobe calls shared by the stages."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FFmpegError(RuntimeError):
    """FFmpeg is missing or a command failed."""


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    width: int | None
    height: int | None
    fps: float | None
    has_audio: bool
    video_start_s: float | None
    audio_start_s: float | None


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FFmpegError(
            f"{name} not found on PATH. Install FFmpeg (winget install Gyan.FFmpeg) "
            "and open a new terminal."
        )
    return path


def probe(path: Path) -> MediaInfo:
    result = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_format",
         "-show_streams", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )  # fmt: skip
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}: {result.stderr.strip()[-500:]}")
    return media_info_from_probe(json.loads(result.stdout))


def media_info_from_probe(data: dict[str, Any]) -> MediaInfo:
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = _float((data.get("format") or {}).get("duration"))
    if duration is None:
        raise FFmpegError("ffprobe reported no duration")
    return MediaInfo(
        duration_s=duration,
        width=video.get("width") if video else None,
        height=video.get("height") if video else None,
        fps=_fraction(video.get("avg_frame_rate")) if video else None,
        has_audio=audio is not None,
        video_start_s=_float(video.get("start_time")) if video else None,
        audio_start_s=_float(audio.get("start_time")) if audio else None,
    )


def run_ffmpeg(args: list[str]) -> None:
    command = [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args]
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise FFmpegError(
            f"ffmpeg exited with {result.returncode}: {result.stderr.strip()[-1000:]}"
        )


def copy_cut(src: Path, dst: Path, start_s: float, duration_s: float) -> None:
    """Cut without re-encoding. The cut starts on the keyframe at or before start_s."""
    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
    try:
        run_ffmpeg(["-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{duration_s:.3f}",
                    "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
                    "-avoid_negative_ts", "make_zero", "-movflags", "+faststart",
                    str(tmp)])  # fmt: skip
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def extract_audio(src: Path, dst: Path, sample_rate: int = 16_000) -> None:
    """Mono 16 kHz WAV: what Whisper expects, and plenty for loudness scoring."""
    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
    try:
        run_ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate),
                    "-c:a", "pcm_s16le", str(tmp)])  # fmt: skip
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _fraction(value: object) -> float | None:
    """ffprobe frame rates look like '60/1' or '0/0'."""
    if not isinstance(value, str) or "/" not in value:
        return _float(value)
    num, _, den = value.partition("/")
    numerator, denominator = _float(num), _float(den)
    if not numerator or not denominator:
        return None
    return numerator / denominator
