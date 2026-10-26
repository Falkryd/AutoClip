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


def run_ffmpeg(args: list[str], cwd: Path | None = None) -> None:
    command = [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args]
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd
    )
    if result.returncode != 0:
        raise FFmpegError(
            f"ffmpeg exited with {result.returncode}: {result.stderr.strip()[-1000:]}"
        )


def keyframe_before(path: Path, t: float, window_s: float = 30) -> float:
    """Time of the last video keyframe at or before t. A stream-copy cut starts there."""
    result = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-read_intervals", f"{max(0.0, t - window_s):.3f}%{t + 0.001:.3f}",
         "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )  # fmt: skip
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}: {result.stderr.strip()[-500:]}")
    keyframe = last_keyframe(result.stdout, t)
    if keyframe is None:
        raise FFmpegError(f"No keyframe within {window_s:.0f} s before {t:.1f} s in {path}")
    return keyframe


def last_keyframe(packets_csv: str, t: float) -> float | None:
    """Parse ffprobe 'pts_time,flags' lines; keyframe packets have K in their flags."""
    times = []
    for line in packets_csv.splitlines():
        pts, _, flags = line.strip().partition(",")
        pts_time = _float(pts)
        if pts_time is not None and "K" in flags and pts_time <= t + 1e-3:
            times.append(pts_time)
    return max(times, default=None)


def measure_loudness(src: Path, start_s: float, duration_s: float, target: str) -> dict[str, str]:
    """First pass of two-pass loudnorm: FFmpeg's measurements of that stretch of audio."""
    result = subprocess.run(
        [_tool("ffmpeg"), "-hide_banner", "-nostdin", "-ss", f"{start_s:.3f}", "-i", str(src),
         "-t", f"{duration_s:.3f}", "-vn", "-af", f"{target}:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )  # fmt: skip
    if result.returncode != 0:
        raise FFmpegError(f"Loudness measurement failed: {result.stderr.strip()[-500:]}")
    return parse_loudnorm(result.stderr)


def parse_loudnorm(stderr: str) -> dict[str, str]:
    """loudnorm prints its measurements as the last JSON object in FFmpeg's log."""
    start, end = stderr.rfind("{"), stderr.rfind("}")
    if start < 0 or end < start:
        raise FFmpegError("loudnorm printed no measurements")
    data = json.loads(stderr[start : end + 1])
    if not {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"} <= data.keys():
        raise FFmpegError(f"Unexpected loudnorm output: {sorted(data)}")
    return data


def encode_to(dst: Path, args: list[str], cwd: Path | None = None) -> None:
    """Run FFmpeg with `args` writing to a temp file next to dst, then move it into place."""
    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
    try:
        run_ffmpeg([*args, str(tmp)], cwd=cwd)
        _replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def copy_cut(src: Path, dst: Path, start_s: float, duration_s: float) -> None:
    """Cut without re-encoding. The cut starts on the keyframe at or before start_s."""
    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
    try:
        run_ffmpeg(["-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{duration_s:.3f}",
                    "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
                    "-avoid_negative_ts", "make_zero", "-movflags", "+faststart",
                    str(tmp)])  # fmt: skip
        _replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def extract_audio(src: Path, dst: Path, sample_rate: int = 16_000) -> None:
    """Mono 16 kHz WAV: what Whisper expects, and plenty for loudness scoring."""
    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
    try:
        run_ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate),
                    "-c:a", "pcm_s16le", str(tmp)])  # fmt: skip
        _replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _replace(tmp: Path, dst: Path) -> None:
    try:
        os.replace(tmp, dst)
    except PermissionError as exc:
        raise FFmpegError(
            f"Can't replace {dst}: another program has it open (a video player?). "
            "Close it and run again."
        ) from exc


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
