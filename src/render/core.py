"""Render each highlight to clips/NN.mp4 (9:16, captions burned in) with a thumbnail."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.common.config import AppConfig, RenderConfig
from src.common.ffmpeg import FFmpegError, encode_to, measure_loudness, probe
from src.common.files import read_json, write_json_atomic, write_text_atomic
from src.common.paths import Artifacts, Target
from src.common.state import utcnow
from src.render.captions import build_ass
from src.render.layout import Layout, filter_graph, plan_layout

_log = logging.getLogger("autoclip.render")

THUMBNAIL_AT = 1 / 3  # thumbnail frame, as a fraction of the clip
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"  # YouTube plays back at about -14 LUFS


class RenderError(RuntimeError):
    """An input is missing, or FFmpeg failed."""


def run_render(cfg: AppConfig, target: Target, *, force: bool = False) -> dict[str, Any]:
    a = target.artifacts
    highlights, words, layout = _inputs(cfg.render, a)
    a.clips_dir.mkdir(parents=True, exist_ok=True)
    rendered = [
        render_clip(clip, words, layout, cfg.render, a, force=force) for clip in highlights["clips"]
    ]
    manifest = {
        "video_id": target.video_id,
        "created_at": utcnow().isoformat(timespec="seconds"),
        "layout": cfg.render.layout,
        "size": [layout.width, layout.height],
        "encoder": cfg.render.video_encoder,
        "clips": rendered,
    }
    write_json_atomic(a.renders, manifest)  # written last: marks the stage done
    if cfg.cleanup.delete_vod_after_render and target.is_pipeline_dir:
        _log.info("Deleting %s (cleanup.delete_vod_after_render is on)", a.vod)
        a.vod.unlink(missing_ok=True)
    return manifest


def render_preview(cfg: AppConfig, target: Target) -> Path:
    """One still of the first clip with the layout and captions, for tuning the crop boxes."""
    a = target.artifacts
    highlights, words, layout = _inputs(cfg.render, a)
    if not highlights["clips"]:
        raise RenderError("highlights.json has no clips to preview")
    clip = highlights["clips"][0]
    a.clips_dir.mkdir(parents=True, exist_ok=True)
    clip_text = clip_words(words, clip["start"], clip["end"])
    captions = _write_captions(a.clips_dir / "layout_preview.ass", clip_text, cfg.render, layout)
    at = clip_text[0]["start"] + 0.1 if clip_text else 2.0  # a moment with a caption on screen
    preview = a.clips_dir / "layout_preview.png"
    args = ["-ss", f"{clip['start']:.3f}", "-i", str(a.vod),
            "-filter_complex", filter_graph(layout, captions), "-map", "[vout]",
            "-ss", f"{at:.2f}", "-frames:v", "1"]  # fmt: skip
    _encode(preview, args, a.clips_dir, cfg.render)
    return preview


def render_clip(
    clip: dict[str, Any],
    words: list[dict[str, Any]],
    layout: Layout,
    rc: RenderConfig,
    a: Artifacts,
    *,
    force: bool = False,
) -> dict[str, Any]:
    name = f"{clip['rank']:02d}"
    video, thumbnail = a.clips_dir / f"{name}.mp4", a.clips_dir / f"{name}.jpg"
    captions = None
    if rc.captions.enabled:
        clip_text = clip_words(words, clip["start"], clip["end"])
        captions = _write_captions(a.clips_dir / f"{name}.ass", clip_text, rc, layout)
    if force or not video.is_file():
        _log.info("Rendering %s: %s (%.0f s)", video.name, clip["title"], clip["duration_s"])
        started = time.monotonic()
        _encode(video, clip_args(a.vod, clip, layout, captions, rc), a.clips_dir, rc)
        _log.info("Rendered %s in %.0f s", video.name, time.monotonic() - started)
    if force or not thumbnail.is_file():
        at = (clip["end"] - clip["start"]) * THUMBNAIL_AT
        _encode(thumbnail, ["-ss", f"{at:.2f}", "-i", str(video), "-frames:v", "1", "-q:v", "3"],
                a.clips_dir, rc)  # fmt: skip
    info = probe(video)
    return {
        "rank": clip["rank"],
        "file": video.name,
        "thumbnail": thumbnail.name,
        "captions": captions,
        "title": clip["title"],
        "reason": clip["reason"],
        "flags": clip.get("flags", []),
        "start": clip["start"],
        "end": clip["end"],
        "stream_start": clip.get("stream_start"),
        "duration_s": round(info.duration_s, 2),
        "width": info.width,
        "height": info.height,
        "size_mb": round(video.stat().st_size / 1e6, 1),
    }


def clip_args(
    vod: Path, clip: dict[str, Any], layout: Layout, captions: str | None, rc: RenderConfig
) -> list[str]:
    # -ss before -i seeks fast, and re-encoding makes the cut frame-accurate anyway.
    duration = clip["end"] - clip["start"]
    args = ["-ss", f"{clip['start']:.3f}", "-i", str(vod), "-t", f"{duration:.3f}",
            "-filter_complex", filter_graph(layout, captions), "-map", "[vout]", "-map", "0:a:0",
            *_video_codec(rc)]  # fmt: skip
    if rc.normalize_audio:
        # Two passes: a single loudnorm pass guesses as it goes and lands 1.5-2.5 LU quiet on
        # short clips, and YouTube only turns loud videos down, never quiet ones up.
        m = measure_loudness(vod, clip["start"], duration, LOUDNORM)
        args += ["-af", f"{LOUDNORM}:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
                        f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
                        f":offset={m['target_offset']}:linear=true"]  # fmt: skip
    return [
        *args,
        "-c:a",
        "aac",
        "-b:a",
        rc.audio_bitrate,
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
    ]


def clip_words(words: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    """The words spoken inside the clip, with times relative to its start."""
    return [
        {"start": round(w["start"] - start, 3), "end": round(min(w["end"], end) - start, 3),
         "word": w["word"]}
        for w in words
        if start <= w["start"] < end
    ]  # fmt: skip


def _inputs(rc: RenderConfig, a: Artifacts) -> tuple[dict[str, Any], list[dict[str, Any]], Layout]:
    for path, stage in (
        (a.highlights, "highlights"),
        (a.transcript, "transcribe"),
        (a.vod, "download"),
    ):
        if not path.is_file():
            raise RenderError(f"{path} not found; run the {stage} stage first")
    highlights = read_json(a.highlights)
    words = [w for s in read_json(a.transcript)["segments"] for w in s["words"]]
    media = probe(a.vod)
    if not media.width or not media.height:
        raise RenderError(f"{a.vod} has no video stream")
    return highlights, words, plan_layout(rc, media.width, media.height)


def _write_captions(
    path: Path, words: list[dict[str, Any]], rc: RenderConfig, layout: Layout
) -> str:
    """Writes the .ass file and returns its bare name (FFmpeg runs inside clips/)."""
    ass = build_ass(words, rc.captions, layout.width, layout.height)
    write_text_atomic(path, ass)
    return path.name


def _video_codec(rc: RenderConfig) -> list[str]:
    if rc.video_encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                "-cq", str(rc.quality), "-b:v", "0", "-profile:v", "high"]  # fmt: skip
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(rc.quality), "-profile:v", "high"]


def _encode(dst: Path, args: list[str], cwd: Path, rc: RenderConfig) -> None:
    try:
        encode_to(dst, args, cwd=cwd)
    except FFmpegError as exc:
        hint = ""
        if rc.video_encoder == "h264_nvenc" and "nvenc" in str(exc).lower():
            hint = " (GPU encoder failed: set render.video_encoder: libx264 to use the CPU)"
        raise RenderError(f"Rendering {dst.name} failed: {exc}{hint}") from exc
