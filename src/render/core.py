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
from src.render.cuts import keep_intervals, kept_duration, remap_words, select_expression
from src.render.layout import Layout, filter_graph, plan_layout

_log = logging.getLogger("autoclip.render")

THUMBNAIL_AT = 1 / 3  # thumbnail frame, as a fraction of the clip
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"  # YouTube plays back at about -14 LUFS


class RenderError(RuntimeError):
    """An input is missing, or FFmpeg failed."""


class _Inputs:
    def __init__(self, rc: RenderConfig, a: Artifacts) -> None:
        for path, stage in (
            (a.highlights, "highlights"),
            (a.transcript, "transcribe"),
            (a.vod, "download"),
        ):
            if not path.is_file():
                raise RenderError(f"{path} not found; run the {stage} stage first")
        self.highlights = read_json(a.highlights)
        transcript = read_json(a.transcript)
        self.words = [w for s in transcript["segments"] for w in s["words"]]
        self.laughs = [s["t"] for s in transcript.get("sounds", []) if s["tag"] == "tertawa"]
        media = probe(a.vod)
        if not media.width or not media.height:
            raise RenderError(f"{a.vod} has no video stream")
        self.layout = plan_layout(rc, media.width, media.height)


def run_render(cfg: AppConfig, target: Target, *, force: bool = False) -> dict[str, Any]:
    a = target.artifacts
    inputs = _Inputs(cfg.render, a)
    a.clips_dir.mkdir(parents=True, exist_ok=True)
    rendered = [
        render_clip(clip, inputs, cfg.render, a, force=force) for clip in inputs.highlights["clips"]
    ]
    manifest = {
        "video_id": target.video_id,
        "created_at": utcnow().isoformat(timespec="seconds"),
        "layout": cfg.render.layout,
        "size": [inputs.layout.width, inputs.layout.height],
        "encoder": cfg.render.video_encoder,
        "jump_cuts": cfg.render.jump_cuts,
        "clips": rendered,
    }
    write_json_atomic(a.renders, manifest)  # written last: marks the stage done
    if cfg.cleanup.delete_vod_after_render and target.is_pipeline_dir:
        _log.info("Deleting %s (cleanup.delete_vod_after_render is on)", a.vod)
        a.vod.unlink(missing_ok=True)
    return manifest


def render_preview(cfg: AppConfig, target: Target) -> Path:
    """One still of the first clip with the layout, hook and captions, for tuning."""
    a = target.artifacts
    inputs = _Inputs(cfg.render, a)
    if not inputs.highlights["clips"]:
        raise RenderError("highlights.json has no clips to preview")
    clip = inputs.highlights["clips"][0]
    a.clips_dir.mkdir(parents=True, exist_ok=True)
    clip_text = clip_words(inputs.words, clip["start"], clip["end"])
    duration = clip["end"] - clip["start"]
    captions = _write_captions(a.clips_dir / "layout_preview.ass", clip_text, clip, duration,
                               cfg.render, inputs.layout)  # fmt: skip
    at = clip_text[0]["start"] + 0.1 if clip_text else 2.0  # a moment with a caption on screen
    preview = a.clips_dir / "layout_preview.png"
    args = ["-ss", f"{clip['start']:.3f}", "-i", str(a.vod),
            "-filter_complex", filter_graph(inputs.layout, captions), "-map", "[vout]",
            "-ss", f"{at:.2f}", "-frames:v", "1"]  # fmt: skip
    _encode(preview, args, a.clips_dir, cfg.render)
    return preview


def render_clip(
    clip: dict[str, Any],
    inputs: _Inputs,
    rc: RenderConfig,
    a: Artifacts,
    *,
    force: bool = False,
) -> dict[str, Any]:
    name = f"{clip['rank']:02d}"
    video, thumbnail = a.clips_dir / f"{name}.mp4", a.clips_dir / f"{name}.jpg"
    source_duration = clip["end"] - clip["start"]
    clip_text = clip_words(inputs.words, clip["start"], clip["end"])
    intervals = None
    if rc.jump_cuts:
        laughs = [t - clip["start"] for t in inputs.laughs if clip["start"] <= t < clip["end"]]
        intervals = keep_intervals(clip_text, laughs, source_duration, rc.max_pause_seconds)
        clip_text = remap_words(clip_text, intervals)
    duration = kept_duration(intervals) if intervals else source_duration
    captions = None
    if rc.captions.enabled or (rc.hook.enabled and clip.get("hook")):
        captions = _write_captions(a.clips_dir / f"{name}.ass", clip_text, clip, duration, rc,
                                   inputs.layout)  # fmt: skip
    if force or not video.is_file():
        _log.info("Rendering %s: %s (%.0f s -> %.0f s)", video.name, clip["title"],
                  source_duration, duration)  # fmt: skip
        started = time.monotonic()
        args = clip_args(a.vod, clip, inputs.layout, captions, rc, intervals)
        _encode(video, args, a.clips_dir, rc)
        _log.info("Rendered %s in %.0f s", video.name, time.monotonic() - started)
    info = probe(video)
    if force or not thumbnail.is_file():
        at = info.duration_s * THUMBNAIL_AT
        _encode(thumbnail, ["-ss", f"{at:.2f}", "-i", str(video), "-frames:v", "1", "-q:v", "3"],
                a.clips_dir, rc)  # fmt: skip
    return {
        "rank": clip["rank"],
        "file": video.name,
        "thumbnail": thumbnail.name,
        "captions": captions,
        "title": clip["title"],
        "hook": clip.get("hook", ""),
        "reason": clip["reason"],
        "flags": clip.get("flags", []),
        "start": clip["start"],
        "end": clip["end"],
        "stream_start": clip.get("stream_start"),
        "source_duration_s": round(source_duration, 2),
        "duration_s": round(info.duration_s, 2),
        "width": info.width,
        "height": info.height,
        "size_mb": round(video.stat().st_size / 1e6, 1),
    }


def clip_args(
    vod: Path,
    clip: dict[str, Any],
    layout: Layout,
    captions: str | None,
    rc: RenderConfig,
    intervals: list[tuple[float, float]] | None = None,
) -> list[str]:
    # -ss and -t before -i: seek fast, and stop reading at the clip's end. After -i, -t only
    # caps the output; with jump cuts the output is shorter, so FFmpeg would decode the VOD
    # to its very end (minutes per clip). Re-encoding makes the cut frame-accurate.
    duration = clip["end"] - clip["start"]
    select = select_expression(intervals) if intervals else None
    args = ["-ss", f"{clip['start']:.3f}", "-t", f"{duration:.3f}", "-i", str(vod),
            "-filter_complex", filter_graph(layout, captions, select), "-map", "[vout]",
            "-map", "0:a:0", *_video_codec(rc)]  # fmt: skip
    # The audio gets exactly the same cuts as the video, so they stay in sync.
    audio_cuts = f"aselect={select},asetpts=N/SR/TB" if select else ""
    if rc.normalize_audio:
        # Two passes: a single loudnorm pass guesses as it goes and lands 1.5-2.5 LU quiet on
        # short clips, and YouTube only turns loud videos down, never quiet ones up.
        pre = f"{audio_cuts}," if audio_cuts else ""
        m = measure_loudness(vod, clip["start"], duration, LOUDNORM, pre_filter=pre)
        args += ["-af", f"{pre}{LOUDNORM}:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
                        f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
                        f":offset={m['target_offset']}:linear=true"]  # fmt: skip
    elif audio_cuts:
        args += ["-af", audio_cuts]
    return [*args, "-c:a", "aac", "-b:a", rc.audio_bitrate, "-ar", "48000",
            "-movflags", "+faststart"]  # fmt: skip


def clip_words(words: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    """The words spoken inside the clip, with times relative to its start."""
    return [
        {"start": round(w["start"] - start, 3), "end": round(min(w["end"], end) - start, 3),
         "word": w["word"]}
        for w in words
        if start <= w["start"] < end
    ]  # fmt: skip


def _write_captions(
    path: Path,
    words: list[dict[str, Any]],
    clip: dict[str, Any],
    duration_s: float,
    rc: RenderConfig,
    layout: Layout,
) -> str:
    """Writes the .ass file (captions and the hook line) and returns its bare name
    (FFmpeg runs inside clips/)."""
    hook = clip.get("hook") if rc.hook.enabled else None
    ass = build_ass(
        words if rc.captions.enabled else [], rc.captions, layout.width, layout.height,
        hook=hook, hook_style=rc.hook, duration_s=duration_s,
    )  # fmt: skip
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
